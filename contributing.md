# Contributing guidelines

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

### How to setup a local server to view the docs

When contibuting to documentation, a good way to check your changes is to view the docs locally. This can be done by setting up a local server. We also recommend docker installation for dev usage.

1. `cd` into the docs folder.
2. Make changes to the documentation.
3. Run `make html`. This will build the documentation locally.
4. Run `make serve` and visit the [port 8000](http://[::]:8000/). You should be able to view the newly built documentation
5. If the changes look good, commit and push the changes to your fork.
6. Make a pull request.

> **Note**
> 
> SoccerTrack is in early development and is not yet ready for production use. We are working on a stable release and will update this README when it is ready. In the meantime, we welcome any feedback or contributions. If you have any questions or something is unclear, please feel free to open an issue.