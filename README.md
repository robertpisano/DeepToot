etup
1. intall pyenv
Mac with homebrew - `brew install pyenv`
Windows with chocolatey - `choco install pyenv-win`

2. install and set version 3.7.0
`pyenv install 3.7.0`
`pyenv global 3.7.0`

3. Add pyenv to your .zshrc/.bashrc 
`profileeval "$(pyenv init --path)"`

3. install pip
`python -m ensurepip --upgrade`

4. install pipenv
`pip install pipenv`

5. install dependencies
`pipenv install`

6. ensure the project root can be found in your pythonPath enviornment variable

7. install postgresql
`brew install postgresql`
`choco install postgresql`

7. update .env variables

