"""Mirror finalized local Phoenix releases to the public GitHub repository."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile


REPO = "MirakSalama/phoenix-xdna1-runtime"
PACK = "phoenix-xdna1-win-x86_64-avx2"
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")
ROOT_NAME = re.compile(r"[0-9]+_bionic_xdna1(?:_[a-z0-9_]+)?")
MANIFEST = "SHA256SUMS.txt"
DOS_NAMES = {"CON","PRN","AUX","NUL"} | {f"{prefix}{i}" for prefix in ("COM","LPT") for i in range(1,10)}


class MirrorError(RuntimeError):
    pass


def fail(message):
    raise MirrorError(message)


def numbered(version):
    if not isinstance(version, str) or not VERSION.fullmatch(version):
        fail(f"Invalid stable version {version!r}")
    return tuple(map(int, version.split(".")))


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def is_link(path):
    return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0)
                                     & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def checked_path(root, relative):
    # Enforce Windows-safe relative names even when validating from a different OS.
    parts = relative.split("/")
    path = root
    for part in parts:
        if (not re.fullmatch(r"[a-zA-Z0-9_. -]+", part) or part in (".", "..")
                or part.endswith((" ", ".")) or part.lower() == ".git"
                or part.split(".")[0].upper() in DOS_NAMES):
            fail(f"Unsafe manifest entry {relative!r}")
        path = path / part
        if is_link(path):
            fail(f"Release contains a link at {relative!r}")
    if not path.is_file():
        fail(f"Missing release file {relative!r}")
    return path


def manifest_entries(data):
    seen = set()
    entries = {}
    for line in data.decode("utf-8-sig").splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})  (.+)", line)
        if not match:
            fail("Invalid SHA256SUMS line")
        digest, name = match.groups()
        if name.casefold() in seen or name.casefold() == MANIFEST.casefold():
            fail(f"Duplicate or recursive manifest entry {name!r}")
        seen.add(name.casefold())
        entries[name] = digest.lower()
    if not entries:
        fail("Empty SHA256SUMS")
    return entries


def discover(experiments):
    versions = {}
    for folder in sorted(experiments.iterdir()):
        if not ROOT_NAME.fullmatch(folder.name) or not folder.is_dir():
            continue
        if is_link(folder):
            fail(f"Numbered release directory is a link: {folder.name}")
        if not (folder/MANIFEST).is_file():
            continue  # Draft or an experiment without a finalized installable distribution.
        manifest = checked_path(folder, MANIFEST).read_bytes()
        entries = manifest_entries(manifest)
        if "install.ps1" not in entries:
            continue
        lock_name = "runtime-lock.json" if "runtime-lock.json" in entries else "compatibility.json"
        if lock_name in entries:
            lock_data = checked_path(folder, lock_name).read_bytes()
            if sha256(lock_data) != entries[lock_name]:
                fail(f"Changed compatibility lock in {folder.name}")
            lock = json.loads(lock_data)
            if lock.get("name" if lock_name == "runtime-lock.json" else "pack") != PACK:
                continue
            version = lock.get("version")
        elif folder.name == "027_bionic_xdna1":
            if "README.md" not in entries or PACK.encode()+b"@0.1.0" not in checked_path(folder, "README.md").read_bytes():
                fail("Unrecognized legacy 0.1.0 release")
            version = "0.1.0"
        else:
            continue
        numbered(version)
        if version in versions:
            fail(f"Conflicting finalized release folders for {version}")
        versions[version] = folder
    return dict(sorted(versions.items(), key=lambda row: numbered(row[0])))


def prepare(version, folder, exporter, staging):
    # Raw inputs are private and temporary. Only the separate sanitized export can reach GitHub.
    staging.mkdir()
    name = f"Phoenix-XDNA1-{version}.zip"
    original = staging / name
    manifest = checked_path(folder, MANIFEST).read_bytes()
    with ZipFile(original, "x", ZIP_DEFLATED) as archive:
        for rel, digest in manifest_entries(manifest).items():
            data = checked_path(folder, rel).read_bytes()
            if sha256(data) != digest:
                fail(f"{folder.name}: private source checksum differs for {rel}")
            archive.writestr(rel, data)
        archive.writestr(MANIFEST, manifest)
    export = staging / "public"
    subprocess.run([sys.executable, "-B", str(exporter), str(original), "--output", str(export)], check=True)
    tree = export / "versions" / version
    asset = export / "assets" / name
    manifest = checked_path(tree, MANIFEST).read_bytes()
    entries = manifest_entries(manifest)
    expected = set(entries) | {MANIFEST}
    actual = set()
    for path in tree.rglob("*"):
        if is_link(path):
            fail("Public export includes a filesystem link")
        if path.is_file():
            actual.add(path.relative_to(tree).as_posix())
    if actual != expected:
        fail("Public source tree differs from the reviewed manifest")
    with ZipFile(asset) as public_zip:
        if len(public_zip.namelist()) != len(expected) or set(public_zip.namelist()) != expected:
            fail("Public ZIP paths differ from the public source tree")
        for rel in expected:
            data = checked_path(tree, rel).read_bytes()
            if rel in entries and sha256(data) != entries[rel]:
                fail(f"Public source checksum differs for {rel}")
            if public_zip.read(rel) != data:
                fail(f"Public source/asset disagreement at {rel}")
    return tree, asset


def git(*args, cwd=None, data=None):
    result = subprocess.run(["git","-c","core.autocrlf=false","-c","core.filemode=false",
                             "-c","credential.https://github.com.helper=",
                             "-c","credential.https://github.com.helper=!gh auth git-credential",
                             "-c","core.hooksPath=NUL" if os.name=="nt" else "core.hooksPath=/dev/null",
                             *args], cwd=cwd, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={**os.environ,"GIT_TERMINAL_PROMPT":"0","GH_PROMPT_DISABLED":"1"})
    if result.returncode:
        fail(f"Git failed: {result.stderr.decode('utf8','replace').strip()}")
    return result.stdout


def gh(*args):
    result = subprocess.run(["gh", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        fail(f"GitHub CLI failed: {result.stderr.decode('utf8','replace').strip()}")
    return result.stdout


def release_index():
    pages = json.loads(gh("api","--hostname","github.com","--paginate","--slurp",
                          f"repos/{REPO}/releases?per_page=100"))
    return {release["tag_name"]:release for page in pages for release in page}


def expected_tree(tree):
    hashes = {}
    for file in tree.rglob("*"):
        if file.is_file():
            data = file.read_bytes()
            hashes[file.relative_to(tree).as_posix()] = hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
    return hashes


def verify_git_tree(checkout, commit, expected):
    hashes = {}
    for line in git("ls-tree","-rz",f"{commit}:runtime",cwd=checkout).split(b"\0"):
        if line:
            info, name = line.split(b"\t", 1)
            mode, kind, oid = info.decode("ascii").split()
            if mode != "100644" or kind != "blob":
                fail("Public runtime tree contains a link, executable bit, or submodule")
            hashes[name.decode()] = oid
    if hashes != expected:
        fail(f"Immutable tag or staged tree differs from the reviewed public source ({commit})")


def tags(checkout):
    return set(git("tag","--list",cwd=checkout).decode().splitlines())


def push_version(checkout, branch, version, tree, expected):
    tag = "v"+version
    existing = [numbered(t[1:]) for t in tags(checkout) if t.startswith("v") and VERSION.fullmatch(t[1:])]
    if existing and numbered(version) <= max(existing):
        fail(f"Missing historical {tag} would regress runtime/; restore it with a reviewed history repair")
    git("read-tree","HEAD",cwd=checkout)
    # Only this temporary clone's runtime/ can be touched. Public root documentation stays in the index.
    dest = checkout/"runtime"
    if dest.exists():
        if is_link(dest) or dest.resolve().parent != checkout.resolve():
            fail("Unexpected temporary runtime path")
        shutil.rmtree(dest)
    shutil.copytree(tree, dest)
    git("add","--force","--all","--","runtime",cwd=checkout)
    staged = git("write-tree",cwd=checkout).decode().strip()
    verify_git_tree(checkout, staged, expected)
    parent = git("rev-parse","HEAD",cwd=checkout).decode().strip()
    commit = git("-c","user.name=Phoenix release mirror","-c","user.email=release-mirror@users.noreply.github.com",
                 "commit-tree", staged, "-p", parent, "-m", f"Release Phoenix XDNA 1 {version}",cwd=checkout).decode().strip()
    git("update-ref",f"refs/heads/{branch}",commit,parent,cwd=checkout)
    git("tag",tag,commit,cwd=checkout)
    # Fast-forward only; GitHub rejects both refs together if another writer wins the race.
    git("push","--atomic","origin",f"HEAD:refs/heads/{branch}",f"refs/tags/{tag}:refs/tags/{tag}",cwd=checkout)


def check_asset(asset_info, local, digest, temp):
    if asset_info["size"] != local.stat().st_size:
        fail("Immutable release asset differs in size")
    remote_digest = asset_info.get("digest")
    if remote_digest:
        if remote_digest.lower() != "sha256:"+digest:
            fail("Immutable release asset SHA-256 differs")
    else:
        # Older GitHub assets can lack a digest. Read only the specific matching-size public asset.
        tmp = temp / "downloaded-release.zip"
        with tmp.open("wb") as f:
            subprocess.run(["gh","api","--hostname","github.com","-H","Accept: application/octet-stream",
                            f"repos/{REPO}/releases/assets/{int(asset_info['id'])}"],stdout=f,check=True)
        if sha256(tmp.read_bytes()) != digest:
            fail("Immutable release asset bytes differ")


def remote_release(release_id):
    return json.loads(gh("api","--hostname","github.com",f"repos/{REPO}/releases/{int(release_id)}"))


def publish_asset(version, asset, digest, release, temp, apply, public_note="", latest=True):
    tag = "v"+version
    if release and release.get("prerelease"):
        fail(f"{tag} already exists as a prerelease")
    if release:
        artifacts = [item for item in release["assets"] if item["name"] == asset.name]
        if len(artifacts) > 1:
            fail(f"{tag} has duplicate assets named {asset.name}")
        if artifacts:
            check_asset(artifacts[0], asset, digest, temp)
    else:
        artifacts = []
    if release and not release["draft"]:
        if not artifacts:
            fail(f"Existing public release {tag} is missing its expected ZIP; refusing an unreviewed repair")
        if digest.lower() not in (release.get("body") or "").lower():
            fail(f"Existing public release {tag} lacks the matching ZIP checksum in its release notes")
        return "verified"
    if release and digest.lower() not in (release.get("body") or "").lower():
        fail("Existing draft lacks the expected release checksum; leaving it untouched")
    if not apply:
        return "would publish"
    if not release:
        body = temp / "release-notes.md"
        intro = public_note or "Verified Phoenix XDNA 1 runtime for Bionic. Installation instructions are in the archive."
        body.write_text(f"{intro}\n\n"
                        f"SHA-256 for `{asset.name}`\n\n`{digest}`\n",encoding="utf8")
        gh("release","create",tag,"--repo",f"github.com/{REPO}","--verify-tag","--draft",
           "--title",f"Phoenix XDNA 1 {version}","--notes-file",str(body))
        release = release_index().get(tag)
        if not release or not release["draft"] or digest.lower() not in (release.get("body") or "").lower():
            fail("Could not confirm the new checksum-bound draft")
    if not artifacts:
        gh("release","upload",tag,str(asset),"--repo",f"github.com/{REPO}")
    current = remote_release(release["id"])
    matches = [item for item in current["assets"] if item["name"]==asset.name]
    if len(matches)!=1 or digest.lower() not in current.get("body","").lower():
        fail("Draft lacks the reviewed ZIP or SHA-256 release note; leaving it unpublished")
    check_asset(matches[0],asset,digest,temp)
    gh("release","edit",tag,"--repo",f"github.com/{REPO}","--draft=false",
       "--latest=true" if latest else "--latest=false")
    current = remote_release(release["id"])
    if current["draft"] or current["prerelease"]:
        fail("GitHub did not expose a stable public release")
    return "published"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments",type=Path,required=True,help="private Phoenix experiments directory")
    parser.add_argument("--apply",action="store_true",help="push verified git refs and publish missing assets")
    parser.add_argument("--exporter",type=Path,default=Path(__file__).with_name("sanitize_release.py"),
                        help="trusted public sanitizer (defaults to sibling script)")
    parser.add_argument("--offline",action="store_true",help="verify local exports without contacting GitHub")
    args = parser.parse_args()
    if args.apply and args.offline:
        parser.error("--offline and --apply are mutually exclusive")
    versions = discover(args.experiments.resolve())
    if not versions:
        fail("No finalized Phoenix releases found")
    with tempfile.TemporaryDirectory(prefix="phoenix-release-") as d:
        temp = Path(d)
        if not args.offline:
            repo = json.loads(gh("api","--hostname","github.com",f"repos/{REPO}"))
            if repo.get("full_name","").casefold()!=REPO.casefold() or repo["private"] or repo["archived"]:
                fail("Wrong, private, or archived GitHub repository")
            branch = repo["default_branch"]
            git("check-ref-format","--branch",branch)
            checkout = temp/"public-repo"
            git("clone","--filter=blob:none","--no-checkout","--branch",branch,
                f"https://github.com/{REPO}.git",str(checkout))
            releases = release_index()
        for version, folder in versions.items():
            staging = temp/("staging-"+version)
            tree, asset = prepare(version,folder,args.exporter.resolve(),staging)
            digest = sha256(asset.read_bytes())
            publisher = args.exporter.resolve().parent
            public_note = ""
            if (publisher/"notes"/f"{version}.md").exists():
                public_note = checked_path(publisher,f"notes/{version}.md").read_text(encoding="utf8").strip()
                if public_note and public_note not in (tree/"README.md").read_text(encoding="utf8"):
                    fail("Reviewed release note differs from the sanitized public README")
            if args.offline:
                print(f"{version}: verified local public export {digest}")
                continue
            expected = expected_tree(tree)
            tag = "v"+version
            if tag in tags(checkout):
                verify_git_tree(checkout,tag,expected)
            elif tag in releases:
                fail(f"Release {tag} exists without its git tag")
            elif args.apply:
                push_version(checkout,branch,version,tree,expected)
            higher_published = any(not r["draft"] and not r["prerelease"] and t.startswith("v")
                                   and VERSION.fullmatch(t[1:]) and numbered(t[1:])>numbered(version)
                                   for t,r in releases.items())
            result = publish_asset(version,asset,digest,releases.get(tag),staging,args.apply,
                                   public_note,latest=not higher_published)
            print(f"{version}: {result} {digest}")


if __name__=="__main__":
    try:
        main()
    except (OSError,ValueError,subprocess.CalledProcessError,MirrorError) as error:
        raise SystemExit(f"Release mirror stopped: {error}")
