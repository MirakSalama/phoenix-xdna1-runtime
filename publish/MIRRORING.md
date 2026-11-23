# Publishing a verified release

The release mirror runs on the Windows host that holds the private Phoenix experiments. It publishes to `MirakSalama/phoenix-xdna1-runtime`. GitHub Actions cannot read those local experiment folders.

## Local release signal

Finish the usual hardware and Bionic checks, then publish the release to its final numbered `NNN_bionic_xdna1...` experiment directory. The existing packaging step writes a `SHA256SUMS.txt` that covers the installer and the versioned compatibility lock. The mirror recognizes this completed distribution. It skips draft folders without a checksum manifest, files outside the manifest, and test packs under nested or working directories. The first historical release uses its known legacy directory and its original runtime identifier.

The manifest checks file integrity. The release owner remains responsible for promoting only verified stable distributions into that directory convention.

## Running the mirror

Install Python 3.10 or newer, Git and the GitHub CLI. The account running the mirror needs an existing `gh` login with permission to push the repository and publish Releases. A scheduled run must use the same authenticated Windows account. Keep the private experiments and the publisher checkout in separate directories.

From the repository root, first run the local verification. Substitute the path to the private experiments directory.

```powershell
python -B publish/mirror_releases.py --experiments "<private-experiments-directory>" --offline
```

To compare the local releases with GitHub without changing the repository, omit `--offline`. To publish or verify all finalized versions, use the following command. A repeat run with matching immutable tags and assets finishes without a push or Release change.

```powershell
python -B publish/mirror_releases.py --experiments "<private-experiments-directory>" --apply
```

For unattended operation, invoke that command after publishing a stable experiment, or have a local scheduler invoke it periodically using absolute script and experiments paths. A nonzero exit means at least one check failed. The command uses a fresh temporary Git checkout and does not depend on a clean developer checkout. No GitHub token is stored in the repository or passed in a remote URL.

## What reaches GitHub

The mirror verifies every file in the private release manifest, reconstructs a temporary input archive, and runs the reviewed, network-free `sanitize_release.py`. Only the sanitized public source tree and its matching ZIP can reach GitHub. The mirror verifies the ZIP bytes against the public file manifest and verifies the Git blob IDs before pushing.

For each new version it replaces `runtime/` on the default branch and tags that commit `vX.Y.Z` in one fast-forward, atomic push. Root documentation and earlier release tags stay available. It creates a draft GitHub Release, uploads one ZIP without overwrite, checks GitHub's SHA-256 digest, and publishes the draft. The release body includes the ZIP checksum. The external asset name is `Phoenix-XDNA1-X.Y.Z.zip`.

The exporter reads optional reviewed release notes from `publish/notes/X.Y.Z.md`. For a new version without notes, it generates installation instructions from the validated version. Set the notes before the first publication. Changing old notes, notices, the sanitizer, or pack files can change output bytes. A mismatch with a public tag or ZIP stops the mirror and leaves the published release intact.

A network interruption after the atomic push can leave a valid tag awaiting its Release. The next run resumes from that tag or a matching checksum-bound draft. If an unrelated draft, asset mismatch, or deleted historical tag is detected, inspect the conflict before retrying. The mirror does not force-push, replace published assets, or delete remote tags.

To run the publisher tests, execute `python -B publish/test_mirror_releases.py -v`. The suite uses a disposable local Git remote and simulated GitHub responses.
