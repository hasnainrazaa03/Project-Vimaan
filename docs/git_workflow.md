# make sure dev is up to date
git checkout dev
git pull origin dev


# create feature branch from dev
git checkout -b feature/branch-name

# work on code (edit files, add new files/folders)


# track ALL changes including new files and folders
git add .


# commit changes
git commit -m "describe your changes"


# push feature branch to GitHub
git push -u origin feature/branch-name


# switch back to dev and sync
git checkout dev
git pull origin dev


# merge feature branch into dev
git merge feature/branch-name


# push merged dev to GitHub
git push origin dev


# delete feature branch locally
git branch -d feature/branch-name


# delete feature branch on GitHub
git push origin --delete feature/branch-name


# clean up stale remote references
git fetch -p
