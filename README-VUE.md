# Frontend with Vue

*This is related to a work in progress, so it is subject to change.*

## Installation

In order to install vue, [node.js](https://nodejs.org/en/) is required. Please check the [documentation](https://nodejs.org/en/download/package-manager/) for your OS.


After node.js is running, Vue can be installed with:
> npm init vue@latest

In order to run the frontend, you must run FastAPI backend with:
> uvicorn api.app:app

end then install all Vue's dependencies in **vue-loki** package.json (`cd vue-loki` first)
> npm install

After installation, node can be run with:
> npm run serve

A symbolic link should be created from a local images' dir inside Vue's public folder, like (MacOS):
> ln -s api/data/img vue-loki/public/data

The frontend enviroment should be avaiable at the defautl local address then:

http://localhost:8080/