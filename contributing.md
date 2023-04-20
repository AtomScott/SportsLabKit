# Contributing guidelines

> **Note**
> 
> SoccerTrack is in early development and is not yet ready for production use. We are working on a stable release and will update this file when it is ready. In the meantime, we welcome any feedback or contributions. If you have any questions or something is unclear, please feel free to open an issue.

We welcome any kind of contribution to our software, from simple comment or question to a full fledged [pull request](https://help.github.com/articles/about-pull-requests/). Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

A contribution can be one of the following cases:

1. you have a question;
1. you think you may have found a bug (including unexpected behavior);
1. you want to make some kind of change to the code base (e.g. to fix a bug, to add a new feature, to update documentation).

The sections below outline the steps in each case.

## You have a question

1. use the search functionality [here](https://github.com/AtomScott/SoccerTrack/issues) to see if someone already filed the same issue;
1. if your issue search did not yield any relevant results, make a new issue;
1. apply the "Question" label; apply other labels when relevant.

## You think you may have found a bug

1. use the search functionality [here](https://github.com/AtomScott/SoccerTrack/issues) to see if someone already filed the same issue;
1. if your issue search did not yield any relevant results, make a new issue, making sure to provide enough information to the rest of the community to understand the cause and context of the problem. Depending on the issue, you may want to include:
    - the [SHA hashcode](https://help.github.com/articles/autolinked-references-and-urls/#commit-shas) of the commit that is causing your problem;
    - some identifying information (name and version number) for dependencies you're using;
    - information about the operating system;
1. apply relevant labels to the newly created issue.

## You want to make some kind of change to the code base

1. (**important**) announce your plan to the rest of the community _before you start working_. This announcement should be in the form of a (new) issue;
1. (**important**) wait until some kind of consensus is reached about your idea being a good idea;
1. if needed, fork the repository to your own Github profile and create your own feature branch off of the latest main commit. While working on your feature branch, make sure to stay up to date with the main branch by pulling in changes, possibly from the 'upstream' repository (follow the instructions [here](https://help.github.com/articles/configuring-a-remote-for-a-fork/) and [here](https://help.github.com/articles/syncing-a-fork/));
1. Install dependencies with `pip3 install -r requirements.txt`;
1. make sure the existing tests still work by running ``pytest``. If project tests fails use ``pytest --keep-baked-projects`` to keep generated project in /tmp/pytest-* and investigate;
1. add your own tests (if necessary);
1. update or expand the documentation;
1. push your feature branch to (your fork of) the Python Template repository on GitHub;
1. create the pull request, e.g. following the instructions [here](https://help.github.com/articles/creating-a-pull-request/).

In case you feel like you've made a valuable contribution, but you don't know how to write or run tests for it, or how to generate the documentation: don't let this discourage you from making the pull request; we can help you! Just go ahead and submit the pull request, but keep in mind that you might be asked to append additional commits to your pull request.

## How to contribute to the documentation

If you would like to contribute to the documentation, you can do so by following the steps below. If you are new to Sphinx, you can read the [reStructuredText Primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) to get started.

1. Fork the repository.
2. Make changes to the documentation.
3. Generate the documentation locally (see below).
4. Make a pull request.

We've also add an action that will build pull requests on readthedocs so you can check the documentation there as well. See below for more information.

### How to setup a local server to view the docs

When contibuting to documentation, a good way to check your changes is to view the docs locally. This can be done by setting up a local server.

1. Run `make html`. This will build the documentation locally.
2. Run `make serve` and visit the [port 8000](http://[::]:8000/). You should be able to view the newly built documentation
3. If the changes look good, commit and push the changes to your fork.

### How to view a preview documentation from PRs

#### Build on pull request events

We create and build a new version when a pull request is opened, and rebuild the version whenever a new commit is pushed.

#### Build status report

Your project’s pull request build status will show as one of your pull request’s checks. This status will update as the build is running, and will show a success or failure status when the build completes.

![](https://docs.readthedocs.io/en/stable/_images/github-build-status-reporting.gif)

<img src='https://raw.githubusercontent.com/AtomScott/SoccerTrack/feature/major_refactor/docs/_static/contributing/docs_01.webp'/>
<img src='https://raw.githubusercontent.com/AtomScott/SoccerTrack/feature/major_refactor/docs/_static/contributing/docs_02.webp'/>

### How to add a new notebook to the documentation

In the `./docs` folder, try to look for `index.rst`. This is the main file that will be used to generate the documentation.  If you want to add a new notebook, you can add it to the `./notebooks` folder and then add the line above to the `index.rst` file. For example, you can add a new notebook by adding the following line to the `index.rst` file:

```rst
..  toctree::
        :maxdepth: 2
        :caption: User Guide
        :hidden:

        notebooks/02_user_guide/new_notebook.ipynb
```

This will add a `new_notebook.ipynb` to the documentation. The notebook itself will be copied from the `./notebooks` folder when `make html` is run, so make sure to add the notebook there. Do not add the notebook to the `./docs/notebooks` folder.

## Try to use optimized images

We all love beautiful images, but they tend to be heavy. We try to keep the size of the repository as small as possible, so please try to use optimized images. You can use [TinyPNG](https://tinypng.com/) to optimize your images.

If you prefer to use a command line tool, you can use [ImageMagick](https://imagemagick.org/index.php) to optimize your images. Here is a insighful [blogpost](https://webdevstudios.com/2022/03/10/quickly-optimize-images/) on how to set up a command in your bash profile! TL;DR:

```bash
function ctwp() {
  extension=$1
  params=$2
      	
  for i in *.$extension
   	do
   	  file=$i
   	  convert $file $params ${file%.*}.webp
   	done
}
```

1. Install ImageMagick CLI
2. Add the above function to a file called .custom_commands.sh in your home directory.
3. Add this to your .bashrc  -> source ~/.custom_commands.sh
4. In the directory full of .jpgs you’d like to convert, run in the terminal: ctwp jpg


## Testing

`poetry run pytest tests`

The Makefile contains the following commands:
```
## format python source code
format:
	poetry run docformatter --in-place -r $(i)
	poetry run black $(i)
	poetry run isort $(i)

## Lint using flake8
lint:
	poetry run docformatter --in-place -r $(i)
	poetry run black $(i)
	poetry run isort $(i)
	poetry run prospector --profile profile.prospector.yaml $(i)

## Run tests using pytest
tests:
	poetry run pytest --cov=./ --cov-report xml 
	/bin/deepsource report --analyzer test-coverage --key python --value-file ./coverage.xml  
```

so you can run `make format` to format the code, `make lint` to lint the code, and `make tests` to run the tests.
