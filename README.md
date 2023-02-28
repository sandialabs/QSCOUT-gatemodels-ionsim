QSCOUT IonSim Gate Models
-------------------------

The [Quantum Scientific Computing Open User Testbed
(QSCOUT)](https://qscout.sandia.gov/) is a five-year DOE program to build a
quantum testbed based on trapped ions that is available to the research
community. As an open platform, it will not only provide full specifications
and control for the realization of all high level quantum and classical
processes, it will also enable researchers to investigate, alter, and optimize
the internals of the testbed and test more advanced implementations of quantum
operations.

This Python package allows [JaqalPaq](https://gitlab.com/jaqal/jaqalpaq)
to use microscopic models of the physical behavior of the QSCOUT hardware
to emulate the native its gate set.

## Installation

The QSCOUT Gate Models package is available on
[GitLab](https://gitlab.com/jaqal/qscout-gatemodels-ionsim).

To install, do **NOT** clone the entire repository unless you are interested
in comparing many old versions of the gates used.  Instead, clone only the
`current` branch:

```bash
git clone --single-branch https://gitlab.com/jaqal/qscout-gatemodels-ionsim.git
cd qscout-gatemodels-ionsim
pip install -e.
```

To perform updates, do a "force" pull.  **Any changes in this repository will
be lost!**  See the development section for a workflow that preserves changes
to the respository.

```bash
git pull -f
```

## Development

The interpolated gates used for these gate models may be 100s of megabytes.
To avoid requiring the download of all of these gates, including very old
ones, the large gate data files are maintained in a separate orphaned branch,
e.g., `data/20211208`.  The much smaller python code files are kept in the
`main-code` branch, which is merged into the data branch.  General
improvements and fixes are merged into `main-code`, which is then merged into
the data branch.  Commits specific to that `data/` branch bypass `main-code`,
and go directly into the appropriate `data/` branch.

As a convenience for users, a `current` branch is maintained.  It should
always track the most recent stable `data/` branch, which means you will not
be able to "fast-forward" when a new set of data files is released.  **Do not
develop on `current`**, the commits will be lost.

As a developer, you may still wish to only download certain branches.  This
can be managed by first cloning the main-code branch:

```bash
git clone --single-branch -b main-code https://gitlab.com/jaqal/qscout-gatemodels-ionsim.git
```

Then, set the `remote.<name>.fetch` git config parameter, which sets the
default for the [git fetch](https://git-scm.com/docs/git-fetch) `<refspec>`
argument, to track the desired upstream branches.  For example, to track the
ionsim gates originally released on December 8, 2021:

```
[remote "origin"]
    url = git@gitlab.com:jaqal/qscout-gatemodels-ionsim.git
    fetch = +refs/heads/main-code:refs/remotes/origin/main-code
    fetch = +refs/heads/data/20211208:refs/remotes/origin/data/20211208
```

Additional fetch lines may be added to track additional branches.  Use
`git ls-remote` to view all available branches.

## License
[Apache 2.0](https://choosealicense.com/licenses/apache-2.0/)

## Questions?

For help and support, please contact
[qscout@sandia.gov](mailto:qscout@sandia.gov).
