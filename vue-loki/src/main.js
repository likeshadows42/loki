import { createApp } from 'vue'
import App from './App.vue'
import axios from 'axios'
// import store from './store'
// import axios from 'axios'

// const app = createApp(App).use(store).use(router)
const app = createApp(App)
app.config.globalProperties.axios=axios


app.mount('#app')