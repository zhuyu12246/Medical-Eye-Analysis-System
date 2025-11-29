const { defineConfig } = require('@vue/cli-service')

module.exports = defineConfig({
  transpileDependencies: true,
  devServer: {
    open: true
  },
  configureWebpack: (config)=>{
    config.devtool = "source-map"
  },
  css: {
    sourceMap: true
  }
})
