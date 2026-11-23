"""Exercise release mirroring against a disposable bare Git remote and simulated GitHub API."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import mirror_releases as mirror


class LocalMirror(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.tree = self.root / "export"
        self.tree.mkdir()
        (self.tree / "README.md").write_bytes(b"verified\r\n")
        self.expected = mirror.expected_tree(self.tree)

    def test_publish_version_preserves_root_and_history(self):
        origin, project = self.root/"origin.git", self.root/"private-clone"
        mirror.git("init","--bare","--initial-branch=main",str(origin))
        mirror.git("init","--initial-branch=main",str(project))
        (project/"README.md").write_bytes(b"root\n")
        (project/"runtime").mkdir()
        (project/"runtime/old.txt").write_bytes(b"earlier\n")
        mirror.git("add",".",cwd=project)
        mirror.git("-c","user.name=Test","-c","user.email=test@example.invalid","commit","-m","First",cwd=project)
        mirror.git("tag","v0.1.0",cwd=project)
        mirror.git("remote","add","origin",str(origin),cwd=project)
        mirror.git("push","--atomic","origin","main","v0.1.0",cwd=project)
        mirror.push_version(project,"main","0.2.0",self.tree,self.expected)
        mirror.verify_git_tree(origin,"v0.2.0",self.expected)
        self.assertEqual(mirror.git("show","main:README.md",cwd=origin),b"root\n")
        self.assertEqual(mirror.git("show","v0.1.0:runtime/old.txt",cwd=origin),b"earlier\n")
        (self.tree/"README.md").write_bytes(b"changed")
        with self.assertRaisesRegex(mirror.MirrorError,"differs"):
            mirror.verify_git_tree(origin,"v0.2.0",mirror.expected_tree(self.tree))
        with self.assertRaisesRegex(mirror.MirrorError,"regress"):
            mirror.push_version(project,"main","0.1.1",self.tree,mirror.expected_tree(self.tree))
        # A competing writer advances main after our fetch. Atomic push must also reject the new tag.
        ahead = mirror.git("-c","user.name=Other","-c","user.email=other@example.invalid","commit-tree",
                           "main^{tree}","-p","main","-m","Concurrent update",cwd=origin).decode().strip()
        mirror.git("update-ref","refs/heads/main",ahead,cwd=origin)
        with self.assertRaises(mirror.MirrorError):
            mirror.push_version(project,"main","0.3.0",self.tree,mirror.expected_tree(self.tree))
        self.assertNotIn("v0.3.0",mirror.tags(origin))
        self.assertEqual(mirror.git("rev-parse","main",cwd=origin).decode().strip(),ahead)

    def test_discovers_new_stable_without_manual_marker(self):
        folder=self.root/"040_bionic_xdna1_future"
        folder.mkdir()
        files={"install.ps1":b"example","compatibility.json":json.dumps(
            {"version":"2.11.3","pack":mirror.PACK}).encode()}
        for name,data in files.items():
            (folder/name).write_bytes(data)
        (folder/mirror.MANIFEST).write_text("".join(f"{mirror.sha256(data)}  {name}\n" for name,data in files.items()))
        (self.root/"041_bionic_xdna1_trials"/"unpublished").mkdir(parents=True)
        self.assertEqual(mirror.discover(self.root),{"2.11.3":folder})

    def test_private_manifest_cannot_escape_or_overwrite(self):
        for rel in ("../outside", "sub/../../escape", ".git/config", "name:stream", "/absolute", "x\\file"):
            with self.subTest(rel=rel), self.assertRaises((mirror.MirrorError,FileNotFoundError)):
                mirror.checked_path(self.tree,rel)
        with self.assertRaisesRegex(mirror.MirrorError,"Duplicate"):
            mirror.manifest_entries(("0"*64+"  x\n"+"1"*64+"  X\n").encode())

    def test_private_checksum_mismatch_before_export(self):
        folder=self.root/"037_bionic_xdna1_stable"
        folder.mkdir()
        (folder/"install.ps1").write_bytes(b"changed")
        (folder/mirror.MANIFEST).write_text("1"*64+"  install.ps1\n",encoding="utf8")
        with patch.object(mirror.subprocess,"run") as external:
            with self.assertRaisesRegex(mirror.MirrorError,"private source checksum differs"):
                mirror.prepare("0.5.0",folder,self.root/"nonexistent.py",self.root/"staging")
            external.assert_not_called()

    def test_existing_release_and_conflict_are_nonmutating(self):
        asset=self.root/"Phoenix-XDNA1-1.0.0.zip"
        asset.write_bytes(b"public only")
        digest=mirror.sha256(asset.read_bytes())
        item={"size":asset.stat().st_size,"digest":"sha256:"+digest,"name":asset.name,"id":7}
        release={"draft":False,"prerelease":False,"assets":[item],"tag_name":"v1.0.0","body":digest}
        with patch.object(mirror,"gh") as github:
            self.assertEqual(mirror.publish_asset("1.0.0",asset,digest,release,self.root,True),"verified")
            item["digest"]="sha256:"+"0"*64
            with self.assertRaisesRegex(mirror.MirrorError,"SHA-256 differs"):
                mirror.publish_asset("1.0.0",asset,digest,release,self.root,True)
            release.update(draft=True,assets=[],body="unrelated developer draft")
            with self.assertRaisesRegex(mirror.MirrorError,"leaving it untouched"):
                mirror.publish_asset("1.0.0",asset,digest,release,self.root,True)
            github.assert_not_called()

    def test_checksum_bound_draft_recovers_after_interrupted_upload(self):
        asset=self.root/"Phoenix-XDNA1-1.0.0.zip"
        asset.write_bytes(b"public only")
        digest=mirror.sha256(asset.read_bytes())
        release={"draft":True,"prerelease":False,"assets":[],"tag_name":"v1.0.0","id":8,"body":digest}
        item={"size":asset.stat().st_size,"digest":"sha256:"+digest,"name":asset.name,"id":7}
        def simulate(*args):
            if args[:2]==("release","upload"):
                self.assertNotIn("--clobber",args)
                release["assets"]=[item]
            elif args[:2]==("release","edit") and "--draft=false" in args:
                release["draft"]=False
            else:
                self.fail(f"Unexpected GitHub action {args}")
        with patch.object(mirror,"gh",side_effect=simulate) as github, patch.object(mirror,"remote_release",return_value=release):
            self.assertEqual(mirror.publish_asset("1.0.0",asset,digest,release,self.root,True),"published")
            self.assertEqual(github.call_count,2)
            self.assertEqual(mirror.publish_asset("1.0.0",asset,digest,release,self.root,True),"verified")
            self.assertEqual(github.call_count,2)


if __name__=="__main__":
    unittest.main()
