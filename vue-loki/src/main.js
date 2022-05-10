import { createApp } from 'vue'
import App from './App.vue'
// import vuetify from './plugins/vuetify'
import { createVuetify } from 'vuetify'
import axios from 'axios'
import { loadFonts } from './plugins/webfontloader'

import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'


const app = createApp(App)
const vuetify = createVuetify({
    components,
    directives,
})


loadFonts()

app.config.globalProperties.axios=axios
app.use(vuetify)
app.mount('#app')