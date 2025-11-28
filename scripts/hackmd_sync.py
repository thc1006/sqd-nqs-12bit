#!/usr/bin/env python3
"""HackMD synchronization script for experiment reports.

Syncs experiment reports and development plans to HackMD for collaboration.

Usage:
    # List all notes
    python scripts/hackmd_sync.py --list

    # Upload experiment report
    python scripts/hackmd_sync.py --upload results/figures/experiment_report.md

    # Update existing note
    python scripts/hackmd_sync.py --update <note_id> --file results/figures/experiment_report.md

    # Download note to local file
    python scripts/hackmd_sync.py --download <note_id> --output output.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import requests


class HackMDAPI:
    """Simple HackMD API client using requests."""

    BASE_URL = "https://api.hackmd.io/v1"

    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    def _request(self, method: str, endpoint: str, data: Optional[dict] = None) -> Any:
        """Make an API request."""
        url = f"{self.BASE_URL}{endpoint}"
        try:
            if method == "GET":
                response = self.session.get(url)
            elif method == "POST":
                response = self.session.post(url, json=data)
            elif method == "PATCH":
                response = self.session.patch(url, json=data)
            elif method == "DELETE":
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unknown method: {method}")

            if response.status_code == 401:
                print("[ERROR] Authentication failed. Check your API token.")
                sys.exit(1)

            if response.status_code >= 400:
                print(f"[ERROR] API error {response.status_code}: {response.text}")
                return None

            if response.status_code in (200, 201, 202, 204):
                if response.status_code == 204 or not response.text:
                    return {}
                try:
                    return response.json()
                except Exception:
                    return {}

            return None

        except requests.RequestException as e:
            print(f"[ERROR] Request failed: {e}")
            return None

    def get_me(self) -> dict:
        """Get current user info."""
        return self._request("GET", "/me")

    def get_note_list(self) -> list:
        """Get list of all notes."""
        return self._request("GET", "/notes") or []

    def get_note(self, note_id: str) -> dict:
        """Get a single note by ID."""
        return self._request("GET", f"/notes/{note_id}")

    def create_note(
        self,
        title: str,
        content: str,
        read_perm: str = "owner",
        write_perm: str = "owner",
    ) -> dict:
        """Create a new note."""
        data = {
            "title": title,
            "content": content,
            "readPermission": read_perm,
            "writePermission": write_perm,
        }
        return self._request("POST", "/notes", data)

    def update_note(
        self,
        note_id: str,
        content: Optional[str] = None,
        read_perm: Optional[str] = None,
        write_perm: Optional[str] = None,
    ) -> dict:
        """Update an existing note."""
        data = {}
        if content is not None:
            data["content"] = content
        if read_perm is not None and write_perm is not None:
            data["readPermission"] = read_perm
            data["writePermission"] = write_perm
        return self._request("PATCH", f"/notes/{note_id}", data)

    def delete_note(self, note_id: str) -> bool:
        """Delete a note."""
        result = self._request("DELETE", f"/notes/{note_id}")
        return result is not None


def get_api_token() -> str:
    """Get HackMD API token from environment."""
    token = os.environ.get("HACKMD_API_TOKEN")

    if not token:
        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("HACKMD_API_TOKEN="):
                        token = line.split("=", 1)[1].strip()
                        break

    if not token:
        print("[ERROR] HACKMD_API_TOKEN not found in environment or .env file")
        print("[INFO] Set your token in .env file or as environment variable")
        sys.exit(1)

    return token


def list_notes(api: HackMDAPI) -> None:
    """List all notes in the account."""
    print("[INFO] Fetching note list...")

    user = api.get_me()
    if user:
        print(f"[INFO] Logged in as: {user.get('name', 'Unknown')}")

    notes = api.get_note_list()

    if not notes:
        print("[INFO] No notes found")
        return

    print(f"\n{'ID':<24} {'Title':<50} {'Last Changed'}")
    print("-" * 90)
    for note in notes:
        note_id = note.get("id", "N/A")
        title = note.get("title", "Untitled")[:48]
        last_changed_raw = note.get("lastChangedAt", "N/A")
        if isinstance(last_changed_raw, int):
            from datetime import datetime
            last_changed = datetime.fromtimestamp(last_changed_raw / 1000).strftime("%Y-%m-%d %H:%M")
        elif isinstance(last_changed_raw, str):
            last_changed = last_changed_raw[:19]
        else:
            last_changed = "N/A"
        print(f"{note_id:<24} {title:<50} {last_changed}")

    print(f"\n[INFO] Total: {len(notes)} notes")


def upload_note(
    api: HackMDAPI,
    file_path: Path,
    title: Optional[str] = None,
    read_perm: str = "owner",
    write_perm: str = "owner",
) -> Optional[str]:
    """Upload a markdown file as a new HackMD note."""
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return None

    content = file_path.read_text(encoding="utf-8")

    if title is None:
        first_line = content.split("\n")[0]
        if first_line.startswith("# "):
            title = first_line[2:].strip()
        else:
            title = file_path.stem

    print(f"[INFO] Uploading: {file_path}")
    print(f"[INFO] Title: {title}")

    result = api.create_note(
        title=title,
        content=content,
        read_perm=read_perm,
        write_perm=write_perm,
    )

    if result:
        note_id = result.get("id")
        publish_link = result.get("publishLink", f"https://hackmd.io/{note_id}")

        print(f"[SUCCESS] Note created!")
        print(f"[INFO] Note ID: {note_id}")
        print(f"[INFO] URL: {publish_link}")

        return note_id

    return None


def update_note(api: HackMDAPI, note_id: str, file_path: Path) -> bool:
    """Update an existing note with new content."""
    if not file_path.exists():
        print(f"[ERROR] File not found: {file_path}")
        return False

    content = file_path.read_text(encoding="utf-8")

    print(f"[INFO] Updating note: {note_id}")
    print(f"[INFO] From file: {file_path}")

    result = api.update_note(note_id=note_id, content=content)

    if result is not None:
        print(f"[SUCCESS] Note updated!")
        print(f"[INFO] URL: https://hackmd.io/{note_id}")
        return True

    return False


def download_note(api: HackMDAPI, note_id: str, output_path: Optional[Path] = None) -> bool:
    """Download a note to a local file."""
    print(f"[INFO] Downloading note: {note_id}")

    note = api.get_note(note_id=note_id)
    if not note:
        return False

    content = note.get("content", "")
    title = note.get("title", "Untitled")

    if output_path is None:
        safe_title = "".join(c if c.isalnum() or c in "._- " else "_" for c in title)
        output_path = Path(f"{safe_title}.md")

    output_path.write_text(content, encoding="utf-8")
    print(f"[SUCCESS] Downloaded to: {output_path}")
    return True


def get_note_info(api: HackMDAPI, note_id: str) -> None:
    """Get detailed information about a note."""
    print(f"[INFO] Fetching note info: {note_id}")

    note = api.get_note(note_id=note_id)
    if not note:
        return

    print(f"\nTitle: {note.get('title', 'Untitled')}")
    print(f"ID: {note.get('id')}")
    print(f"Short ID: {note.get('shortId')}")
    print(f"Created: {note.get('createdAt')}")
    print(f"Last Changed: {note.get('lastChangedAt')}")
    print(f"Read Permission: {note.get('readPermission')}")
    print(f"Write Permission: {note.get('writePermission')}")
    print(f"Publish Link: {note.get('publishLink')}")

    content = note.get("content", "")
    print(f"Content Length: {len(content)} characters")


def sync_reports(api: HackMDAPI) -> None:
    """Sync all experiment reports to HackMD."""
    reports = [
        ("results/figures/experiment_report.md", "NQS-SQD 12-bit Experiment Report"),
        ("DEVELOPMENT_PLAN.md", "NQS-SQD Development Plan"),
    ]

    project_root = Path(__file__).parent.parent

    for rel_path, title in reports:
        file_path = project_root / rel_path
        if file_path.exists():
            print(f"\n{'='*60}")
            upload_note(api, file_path, title=title)
        else:
            print(f"[WARN] File not found: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="HackMD synchronization for NQS-SQD project")
    parser.add_argument("--list", action="store_true", help="List all notes")
    parser.add_argument("--upload", type=str, help="Upload a markdown file as new note")
    parser.add_argument("--update", type=str, help="Update existing note by ID")
    parser.add_argument("--download", type=str, help="Download note by ID")
    parser.add_argument("--info", type=str, help="Get note info by ID")
    parser.add_argument("--file", type=str, help="File path for update operation")
    parser.add_argument("--output", type=str, help="Output path for download")
    parser.add_argument("--title", type=str, help="Custom title for upload")
    parser.add_argument("--sync-all", action="store_true", help="Sync all reports to HackMD")
    parser.add_argument("--delete", type=str, help="Delete note by ID")

    args = parser.parse_args()

    token = get_api_token()
    api = HackMDAPI(token)

    if args.list:
        list_notes(api)
    elif args.upload:
        upload_note(api, Path(args.upload), title=args.title)
    elif args.update:
        if not args.file:
            print("[ERROR] --file required for update operation")
            sys.exit(1)
        update_note(api, args.update, Path(args.file))
    elif args.download:
        output = Path(args.output) if args.output else None
        download_note(api, args.download, output)
    elif args.info:
        get_note_info(api, args.info)
    elif args.sync_all:
        sync_reports(api)
    elif args.delete:
        confirm = input(f"Delete note {args.delete}? [y/N]: ")
        if confirm.lower() == "y":
            if api.delete_note(args.delete):
                print("[SUCCESS] Note deleted")
            else:
                print("[ERROR] Failed to delete note")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
