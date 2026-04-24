"""
upload_to_github.py — Einzeiler Upload auf GitHub (privates Repo)

Nutzung:
  1. Token holen: https://github.com/settings/tokens
     - 'Generate new token (classic)'
     - Scope: 'repo' ankreuzen
     - Generate & kopieren
  2. Script ausfuehren:
     python upload_to_github.py --token ghp_DEINTOKEN
"""

import argparse
import json
import subprocess
import sys
import urllib.request
import urllib.error

REPO_NAME = 'dbd-survivor-detector'
DESCRIPTION = 'YOLOv8-based real-time survivor detection for Dead by Daylight'
GITHUB_USER = 'Pommes2030450'


def create_repo(token: str, private: bool = True):
    data = json.dumps({
        'name': REPO_NAME,
        'description': DESCRIPTION,
        'private': private,
        'has_issues': True,
        'has_projects': False,
        'has_wiki': False,
        'auto_init': False,
    }).encode()

    req = urllib.request.Request(
        'https://api.github.com/user/repos',
        data=data,
        headers={
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json',
            'User-Agent': 'dbd-uploader'
        },
        method='POST'
    )

    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            print(f"Repository erstellt: {result['html_url']}")
            return result['clone_url'], result['ssh_url']
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        if e.code == 422 and 'already exists' in body:
            print(f"Repository existiert schon — nutze bestehendes")
            return (f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git",
                    f"git@github.com:{GITHUB_USER}/{REPO_NAME}.git")
        print(f"Fehler ({e.code}): {body}")
        sys.exit(1)


def run_git(cmd, check=True, env=None):
    """Führt einen git-Befehl aus."""
    print(f"  $ git {' '.join(cmd)}")
    result = subprocess.run(
        ['git'] + cmd,
        capture_output=True, text=True, env=env
    )
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        if check:
            sys.exit(1)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--token', required=True, help='GitHub Personal Access Token')
    p.add_argument('--public', action='store_true', help='Oeffentlich statt privat')
    args = p.parse_args()

    if not args.token.startswith(('ghp_', 'github_pat_')):
        print("Warnung: Token sieht ungewöhnlich aus (erwartet ghp_... oder github_pat_...)")

    # 1. Repo erstellen
    print(f"\n[1/4] Repository erstellen ({'public' if args.public else 'PRIVATE'})...")
    https_url, ssh_url = create_repo(args.token, private=not args.public)

    # Token in URL einbauen fuer Push ohne Passwort-Abfrage
    push_url = https_url.replace('https://',
                                 f'https://{GITHUB_USER}:{args.token}@')

    # 2. Remote setzen (oder ueberschreiben)
    print(f"\n[2/4] Remote konfigurieren...")
    run_git(['remote', 'remove', 'origin'], check=False)
    run_git(['remote', 'add', 'origin', push_url])

    # 3. Branch-Name setzen
    print(f"\n[3/4] Branch auf 'main' setzen...")
    run_git(['branch', '-M', 'main'])

    # 4. Pushen
    print(f"\n[4/4] Push zu GitHub...")
    run_git(['push', '-u', 'origin', 'main'])

    # Remote URL zurueck ohne Token setzen (falls jemand das Repo shared)
    run_git(['remote', 'set-url', 'origin', https_url])

    print(f"\nFERTIG! Repository online:")
    print(f"   https://github.com/{GITHUB_USER}/{REPO_NAME}")


if __name__ == '__main__':
    main()
