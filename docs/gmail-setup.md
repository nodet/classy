# Gmail API Setup Guide

This guide walks through every step needed to connect this project to your Gmail account.

## Overview

Google requires an "app" to access Gmail on your behalf. You create this app (free), authorize it once in your browser, and then the scripts can read your email without further interaction.

The pieces:
- **Google Cloud project** — a container for the "app" (costs nothing)
- **OAuth client credentials** — a file (`client_secret.json`) that identifies your app
- **Token** — generated when you authorize; stored locally for reuse

---

## Step 1: Create a Google Cloud project

1. Go to https://console.cloud.google.com/
2. Sign in with the Google account whose Gmail you want to classify.
3. At the top bar, click the project dropdown (may say "Select a project").
4. Click **New Project**.
5. Name it something like `gmail-classifier`. Organization can be "No organization".
6. Click **Create**. Wait a few seconds.
7. Make sure the new project is selected in the top dropdown.

---

## Step 2: Enable the Gmail API

1. In the left sidebar, go to **APIs & Services → Library** (or search "Gmail API" in the top search bar).
2. Find **Gmail API** and click on it.
3. Click **Enable**.

---

## Step 3: Configure the OAuth consent screen

This tells Google what your "app" is. Since only you will use it, choose "External" (counterintuitively, it's simpler for personal use).

1. Go to **APIs & Services → OAuth consent screen**.
2. Choose **External**. Click **Create**.
3. Fill in:
   - App name: `gmail-classifier`
   - User support email: your email
   - Developer contact email: your email
4. Click **Save and Continue**.
5. On the "Scopes" page, click **Add or Remove Scopes**.
   - Search for `gmail.readonly`
   - Check the box for `https://www.googleapis.com/auth/gmail.readonly`
   - Click **Update**, then **Save and Continue**.
6. On the "Test users" page, click **Add Users**.
   - Enter your own Gmail address.
   - Click **Add**, then **Save and Continue**.
7. Click **Back to Dashboard**.

> **Note:** Your app will stay in "Testing" mode. That's fine — it means only the test users you listed (yourself) can authorize it. No need to "publish" it.

---

## Step 4: Create OAuth credentials

1. Go to **APIs & Services → Credentials**.
2. Click **+ Create Credentials → OAuth client ID**.
3. Application type: **Desktop app**.
4. Name: `gmail-classifier` (or anything).
5. Click **Create**.
6. A dialog appears with your client ID and secret. Click **Download JSON**.
7. Rename the downloaded file to `client_secret.json`.
8. Move it to the `credentials/` directory in this project:

```
mv ~/Downloads/client_id_*.json credentials/client_secret.json
```

---

## Step 5: Authorize (first run)

Run:

```
make fetch-training
```

This will:
1. Open your default browser to a Google sign-in page.
2. You'll see a warning "This app isn't verified" — click **Continue** (it's your own app).
3. Grant permission to read your email.
4. The browser will show "The authentication flow has completed."
5. Back in the terminal, the script starts fetching messages.

After this, a `credentials/token.json` file is created. Future runs won't need the browser.

---

## Step 6: Fetch training data

```
make fetch-training
```

This fetches all messages from your user-created Gmail labels and stores them in `data/training.db`.

To fetch only specific labels:

```
uv run python scripts/fetch_training_data.py --labels Technology Newsletters Travel
```

---

## Step 7: Fetch inbox sample (for dry-run classification)

```
make fetch-inbox
```

This fetches the 100 most recent inbox messages and stores them in `data/inbox_sample.db`.

To fetch more:

```
uv run python scripts/fetch_inbox.py --count 200
```

---

## Files created

```
credentials/
  client_secret.json   # Your OAuth app identity (do NOT commit)
  token.json           # Your authorization token (do NOT commit)
data/
  training.db          # Labeled messages from Gmail
  inbox_sample.db      # Recent inbox messages for dry-run
```

Both `credentials/` and `data/` are in `.gitignore`.

---

## Troubleshooting

**"Access blocked: This app's request is invalid"**
- Make sure you added yourself as a test user in the consent screen (Step 3.6).

**"File not found: credentials/client_secret.json"**
- You need to download the OAuth credentials (Step 4.6) and place the file there.

**Token expired after long inactivity**
- Delete `credentials/token.json` and run again — it will re-authorize.

**"Quota exceeded" or 429 errors**
- The Gmail API has rate limits. Wait a minute and retry. For very large mailboxes, the script may need multiple runs (it skips already-fetched messages).
