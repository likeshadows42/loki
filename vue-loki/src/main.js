import { createApp } from 'vue'
import App from './App.vue'
import vuetify from './plugins/vuetify'
import axios from 'axios'
import { loadFonts } from './plugins/webfontloader'

loadFonts()

const app = createApp(App)
app.config.globalProperties.axios=axios
app.use(vuetify)
app.mount('#app')