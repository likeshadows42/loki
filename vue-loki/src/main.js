import { createApp } from 'vue'
import { createStore } from 'vuex'
import App from './App.vue'
// import vuetify from './plugins/vuetify'
import { createVuetify } from 'vuetify'
import axios from 'axios'
import { loadFonts } from './plugins/webfontloader'

import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'



const vuetify = createVuetify({
    components,
    directives,
})

// Create a new store instance.
const store = createStore({
    state () {
      return {
        $showHidden: false
      }
    },

    mutations: {
        hideUnhide(state) {
          state.$showHidden = !state.$showHidden
        }
      }
  })
  

loadFonts()

const app = createApp(App)
app.config.globalProperties.axios=axios
// app.config.globalProperties.$showHidden = true

app.use(vuetify)
app.use(store)
app.mount('#app')