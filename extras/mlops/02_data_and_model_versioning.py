# data and model versioning solves the thing git can't. git is built for text and
# chokes on multi-gigabyte weights and datasets. dvc keeps the big files in separate
# storage and puts small pointer files in git, so a git commit now pins not just your
# code but the exact data and weights that produced a result. that lineage is what lets
# you answer "what data made this model?" a year later.
#
# dvc is mostly a command line tool that sits next to git, so this lesson is as much
# about the cli workflow as about python. we use subprocess to drive it and reason about
# what each command does to the repo.

# LEARNING OBJECTIVES:
# - Explain why git alone fails on large binary data and how dvc's pointer model fixes it
# - Track a dataset or checkpoint with dvc and see the tiny .dvc pointer git actually commits
# - Understand a remote (local dir, s3, gdrive) as where the real bytes live
# - Reproduce an old result by checking out an old commit and pulling its matching data
# - Connect a data version to an experiment run so a metric traces back to exact inputs
# - See how this pairs with lesson 01: track logs the run, dvc versions what it ran on

import subprocess
from pathlib import Path


def main():
    # step 1: dvc init in the repo and note the .dvc dir and gitignore entries it creates
    # step 2: dvc add a large file (a dataset or checkpoint) and inspect the .dvc pointer file
    # step 3: git add/commit the pointer, confirm the big file itself is gitignored
    # step 4: configure a dvc remote (start with a local folder) and dvc push the bytes there
    # step 5: delete the local copy, dvc pull it back, prove the pointer is enough to restore it
    # step 6: check out an earlier commit + dvc checkout and watch the data snap to that version
    pass


if __name__ == "__main__":
    main()
