# ab_tools
A simple description of the repo

## Getting Started
## Prerequisites
## Installation
### step 1. create the project with cruft
Create the repo using cruft
```
cruft create https://github.com/abuttonCH/ab-cookiecutter.git
```

You will then be prompted to provide the proejct and author name, as well as a brief description of the project

### step 2. Activate the virtual environment and install all packages.
Activated poetry shell and install the packages
```
poetry shell
poetry install
```

### step 3. Create the remote
After creating the local repository you then need to create a remote repo on github. This should have the same project name as the local repo.

Once you've created the remote repo run the following git commands
```
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/GITHUB_ACCOUNT/ab_tools.git
git push origin main 
```

### step 4. Check the initial pre-commits and tests
As a last step, check that the pre-commits and tests pass for the initial repo
```
pre-commit run -a
pytest
```
Both of these should pass. If they didn't, that likely means that something is wrong with the cookiecutter template.
## Usage
This section should demonstrate simple usage of the project.

For example
```
python ab_tools.main
```
## License
This is where you specify the license information.
## Authors
Alex Button
## Acknowledgements
Here you can put any acknowledgements to other people or work that supported this project. 
