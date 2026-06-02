"""OAuth2 authentication for Gmail API."""

import os
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/pubsub",
]

DEFAULT_CREDENTIALS_DIR = Path("credentials")
TOKEN_FILE = "token.json"
CLIENT_SECRETS_FILE = "client_secret.json"


def get_gmail_service(credentials_dir: Path = DEFAULT_CREDENTIALS_DIR):
    """Authenticate and return a Gmail API service object.

    On first run, opens a browser for OAuth consent.
    On subsequent runs, uses the stored refresh token.
    """
    token_path = credentials_dir / TOKEN_FILE
    secrets_path = credentials_dir / CLIENT_SECRETS_FILE

    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not secrets_path.exists():
                raise FileNotFoundError(
                    f"OAuth client secrets not found at {secrets_path}. "
                    "Download from Google Cloud Console and place there."
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets_path), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save token for next run
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)
