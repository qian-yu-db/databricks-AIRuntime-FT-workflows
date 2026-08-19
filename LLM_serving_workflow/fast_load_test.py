"""Locust locustfile for load-testing a Databricks custom-LLM serving endpoint.

Rebased onto the official Databricks example
(https://docs.databricks.com/aws/en/machine-learning/model-serving/configure-load-test):

  * FastHttpUser (geventhttpclient) — ~5-6x more requests/core than HttpUser, so the
    client can out-scale the endpoint on the recommended >=32-core node.
  * Service-principal OAuth token (plain client_credentials, scope=all-apis) — the correct
    flow for a NON route-optimized endpoint — refreshed on a fixed lifetime.

Measures whole-request latency (Locust's built-in timer). For the per-token view (TTFT,
time-per-output-token), read the server-side vLLM histograms in serve_custom_llms_metrics.py
rather than measuring them client-side — FastHttpUser's response object has no streaming
line iterator, so client-side TTFT is not supported here.

This is NOT a Databricks notebook — it is a plain python module that the `locust` process
imports and runs. The driver notebook (locust_load_test.py) sets the env vars below and
launches locust via the CLI, pointing --locustfile at this file.

Environment variables (set by the driver notebook):
  DATABRICKS_WORKSPACE_URL   e.g. https://<host>
  DATABRICKS_ENDPOINT_NAME   serving endpoint name
  CLIENT_ID / CLIENT_SECRET  service principal OAuth creds
  INPUT_JSON                 path to the request payload (default: input.json)
"""

import os
import json
from datetime import datetime, timedelta
from typing import Tuple

import requests
from locust import FastHttpUser, task


class LoadTestUser(FastHttpUser):
    def get_oauth_token(
        self, token_lifetime: timedelta = timedelta(minutes=55)
    ) -> Tuple[str, datetime]:
        """Fetch a workspace OAuth token for the service principal.

        Plain `client_credentials` grant with `scope=all-apis` — the correct flow for a
        NON route-optimized endpoint (hit via `serving-endpoints/<name>/invocations`).
        (The scoped `authorization_details` / `query_inference_endpoint` grant is only for
        route-optimized endpoints; using it here yields tokens the gateway rejects with 401.)

        Returns the access token and its expiration datetime.
        """
        response = requests.post(
            url=f"{self.workspace_url}/oidc/v1/token",
            auth=(self.CLIENT_ID, self.CLIENT_SECRET),
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "client_credentials",
                "scope": "all-apis",
            },
        )
        # Surface token failures loudly instead of silently sending `Bearer None`
        # (which is what produced 100% HTTP 401s).
        response.raise_for_status()
        body = response.json()
        if "access_token" not in body:
            raise RuntimeError(f"OAuth token request failed: {body}")
        return body["access_token"], datetime.now() + token_lifetime

    def on_start(self):
        # Read the request payload from disk (INPUT_JSON, defaulting to input.json in cwd).
        with open(os.environ.get("INPUT_JSON", "input.json"), "r") as json_features:
            self.model_input = json.load(json_features)
        # Load configuration from the environment (set by the driver notebook).
        self.endpoint_name = os.environ.get("DATABRICKS_ENDPOINT_NAME")
        self.CLIENT_ID = os.environ.get("CLIENT_ID")
        self.CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
        self.workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL")
        self.oauth, self.expiration = self.get_oauth_token()

    def check_token_expiration(self):
        """Refresh the OAuth token if it has expired."""
        if datetime.now() > self.expiration:
            self.oauth, self.expiration = self.get_oauth_token()

    @task
    def query_single_model(self):
        self.check_token_expiration()
        headers = {"Authorization": f"Bearer {self.oauth}"}
        path = f"serving-endpoints/{self.endpoint_name}/invocations"
        # Locust's own timer captures total response time.
        self.client.post(path, headers=headers, json=self.model_input)
