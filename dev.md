---
layout: inner
title: Dev
permalink: /dev/
---
## Contributing

This document aims to direct you to further instructions for contributing to SoccerTrack.
If you have any questions, feel free to reach out to [Atom Scott](https://atomscott.me/), who is the the primary maintainer.

### Dataset

Please [create an issue](https://github.com/AtomScott/SoccerTrack/issues) with the appropriate template. 

### Algorithm / Package

We welcome any bug reports, feature requests, documentation contributions etc. If you are thinking of contributing, please start by reading the [contributor guide in our read the docs](https://soccertrack.readthedocs.io/en/latest/) page!

### This Webapge

Found a typo? Think you can improve this page? [Send a pull request]() by following these steps:


#### 1. First, setup the webpage in a local environment

* Install Jekyll, Ruby and Bundler (see [here](https://docs.github.com/en/pages/setting-up-a-github-pages-site-with-jekyll/testing-your-github-pages-site-locally-with-jekyll#prerequisites)).

* Clone the repo

 ```
 git clone https://github.com/AtomScott/SoccerTrack.git
 ```

* Run the following commands

 ```
 bundle install
 bundle exec jekyll serve
 # Navigate to http://localhost:4000 in your browser.
 ```


#### 2. Write code & confirm it works locally

* Create a branch and prepare your changes. Don't work on `gh-pages` branch directly as it may become complicatied when you need to rebase.

* If you completed step 1. correctly, then most of the changes should be reflected without needing to restart the jekyll server. Reload localhost:4000 and if it looks good you should be ok. 

* Note that some files will affect how github serves this webpage. Try not to change those files since it will cause discrepancy between what you see locally and here on github.

#### 3. Make a pull request

* Make sure your pull request has a good name! It will be useful later when you may work on multiple tasks/PRs.

* See the Github Docs on [pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) or [creating a PR](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) for further details.
