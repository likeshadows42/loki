const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  devServer: {
    proxy: {
      '/api/fr': {
        target: 'http://127.0.0.1:8000/'
      },
    }
  },

  transpileDependencies: true,

  pluginOptions: {
    vuetify: {
			// https://github.com/vuetifyjs/vuetify-loader/tree/next/packages/vuetify-loader
		}
  },

  // publicPath: 
  //   process.env.NODE_ENV === 'build'
  //     ? './'
  //     : 'http://localhost:8080/'
})
