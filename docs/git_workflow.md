# make sure main is up to date
git checkout main
git pull origin main


# create feature branch from main
git checkout -b feature/branch-name

# work on code (edit files, add new files/folders)


# track ALL changes including new files and folders
git add .


# commit changes
git commit -m "describe your changes"


# push feature branch to GitHub
git push -u origin feature/branch-name


# switch back to main and sync
git checkout main
git pull origin main


# merge feature branch into main
git merge feature/branch-name


# push merged main to GitHub
git push origin main


# delete feature branch locally
git branch -d feature/branch-name


# delete feature branch on GitHub
git push origin --delete feature/branch-name


# clean up stale remote references
git fetch -p
