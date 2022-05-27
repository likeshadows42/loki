<script>

// imports
import Home from './components/home.vue'
import compPeopleList from './components/people-list.vue'
import compImgUngrouped from './components/img-ungrouped-show.vue'
import compGroupsShow from './components/img-groups-show.vue'
import ImagesVerifiedWithUpload from './components/img-verifier-with-upload.vue'
import ImagesVerifiedWithoutUpload from './components/img-verifier-without-upload.vue'
import ImagesUploaderAdv from './components/img-uploader-adv.vue'
import zipUploader from './components/zip-uploader.vue'
import axios from "axios"

// exports
export default {
  name: 'App',

  components: {
    Home,
    compPeopleList,
    compImgUngrouped,
    compGroupsShow,
    ImagesVerifiedWithUpload,
    ImagesVerifiedWithoutUpload,
    ImagesUploaderAdv,
    zipUploader
  },

  data() {
    return {
      MainContent: null,
      MainContentRaw: null,
      compPeopleListToggler: null,
      imgUploaderWithUpToggler: false,
      imgUploaderWithoutUpToggler: false,
      imgUploaderAdvToggler: false,
      zipUploaderToggler: false,
      compImgUngroupedToggler: false,
      compGroupsToggler: true,
    }
  },

  methods: {
    async axiosGet(url) {
      try {
        const res = await axios.get(url)
        // console.log(res.data)
        return res.data
      } catch(error) {
        console.log(error)
      }
    },

    async axiosPost(url, params={}) {
      try {
        const res = await axios.post(url, params)
        // console.log(res.data)
        return res.data
      } catch(error) {
        console.log(error)
      }
    },

    async getGlobaldata() {
      console.log("Getting global parameters")
      this.MainContent = await this.axiosGet('/api/fr/debug/inspect_globals')
    },
    
    async dbLoadFromDir() {
      this.mainClear()
      console.log("Loading database from directory")
      const params = {"force_create": true}
      this.MainContentRaw = '<h2>Load images from directory</h2><p>Wait until the process will be completed</p>'
      const result = await this.axiosPost('/api/fr/faces/import_from_directory', params)
      this.MainContentRaw += '<p><b>' + await result.message + '</b></p>'
    },

    async imgdirClear() {
      console.log("Clearing default image dir")
      this.mainClear()
      const result = await this.axiosPost('/api/fr/utility/clear_image_dir')
      this.MainContent = result.message
    },

    async dbClear() {
      console.log("Clearing database")
      this.mainClear()
      const result = await this.axiosPost('/api/fr/utility/clear_database')
      this.MainContent = result.message
    },

    async dbReload() {
      console.log("Reload database")
      this.MainContent = await this.axiosPost('/api/fr/database/reload')
    },

    async serverReset(){
      console.log("Reset server&restart")
      const params = {"no_database": true}
      this.MainContent = await this.axiosPost('/api/fr/debug/reset_server', params)
    },

    mainClear() {
      this.MainContent = null
      this.MainContentRaw = null
      this.imgUploaderWithUpToggler = false
      this.imgUploaderWithoutUpToggler = false
      this.imgUploaderAdvToggler = false
      this.zipUploaderToggler = false
      this.MainContentRaw = null
      this.compImgUngroupedToggler = false
      this.compGroupsToggler = false
    },

    sectionToggler(section) {
      const components = ['MainContent','MainContentRaw','compPeopleListToggler','imgUploaderWithUpToggler','imgUploaderWithoutUpToggler','imgUploaderAdvToggler','zipUploaderToggler','compImgUngroupedToggler','compGroupsToggler']
      for(var comp of components) {
          this[comp] = (comp == section ? true : null)
        }
    },

    handleImages(files) {
      console.log(files)
    },
    
    testLog(evt) {
      // console.log("it works!")
      console.log(evt)
    },

    imgOnSubmit(msg) {
      console.log("Img received!")
      console.log(msg[0])
      const imgBase = '/data/'
      const imgURLs = msg[0].image_names
      let imgs = ''
      for (const imgURL of imgURLs) {
        imgs += '<img @click="testLog(1)" src="'+ imgBase + imgURL +'" class="img_thumb"/>'
      }
      
      // imgs = `
      // <div v-for="imgURL in imgURLs">
      //   <img src="{{ imgBase }}{{imgURL}}" class="img_thumb"/>
      // </div>
      // `

      // let output = '<div class="container"><div class="row">' + imgs + '</div></div>'

      this.MainContentRaw = imgs
    },

    zipOnSubmit(msg) {
      console.log(msg)
      // this.sectionToggler('zipUploaderToggler')
      // this.MainContent = null
      // this.MainContentRaw = "<p><b>All done!</b></p>"
      // this.MainContentRaw += '<p>' + msg.message + '</p>'
      // this.MainContentRaw += '<p>New images: <b>' + msg.n_records + '</b></p>'
      // this.MainContentRaw += '<p>Skipped files: <b>' + msg.n_skipped + '</b></p>'

      // this.MainContent = msg
    },


  },

  created() {
    // this.const_test1 = 'ok';
  },

  mounted() {
    // console.log(`Constant "const_test1"= "${this.const_test1}"`)
  },
}
</script>


<template>
  <div class="wrapper">
    <header class="header"><Home msg="Loki MVP"/></header>
    <aside class="aside aside-1">
      
      <h3>Face recognition</h3>
      <p><a href="#" @click.prevent="sectionToggler('compPeopleListToggler')">Show people</a></p> 
      <p><a href="#" @click.prevent="sectionToggler('compGroupsToggler')">Show people (detail)</a></p>
      <p><a href="#" @click.prevent="sectionToggler('compImgUngroupedToggler')">Show ungrouped images</a></p>
      <!-- <p><a href="#" @click.prevent="imgUploaderAdvToggle">Upload multiple images</a></p> -->
      
      <h3>Load images</h3>
      <p><a href="#" @click.prevent="sectionToggler('zipUploaderToggler')">Load images from zip</a></p>
      <!-- <p><a>Upload multiple images</a></p> -->

      <h3>Verify images</h3>
      <p><a href="#" @click.prevent="sectionToggler('imgUploaderWithoutUpToggler')">Verify image (without upload)</a></p>
      <!-- <p><a href="#" @click.prevent="imgVerWithUpToggle">Verify image (with upload)</a></p> -->
      
      <!-- <p><a>Verify image (with upload)</a></p> -->
      <!-- <SecondComp @response="(msg) => MainContent = msg"/>
      <SecondComp @response="mainClear"/> -->

      

      <h3>Utility</h3>
      <!-- <p><a href="#" @click.prevent="">Load database</a></p>
      <p><a href="#" @click.prevent="">Save database</a></p> -->
      <p><a href="#" @click.prevent="dbLoadFromDir">Load images from default directory</a></p>
      <p><a href="/data/" target="_blank">Show img_dir content</a></p>
      <p><a href="#" @click.prevent="dbClear">Clear database</a></p>
      <p><a href="#" @click.prevent="imgdirClear">Clear image dir</a></p>
      <!-- <p><a href="#" @click.prevent="dbReload">Reload database</a></p> -->

      <!-- <h3>Utility</h3>
      <p><a href="#" @click.prevent="getGlobaldata">Get global parameters</a></p>
      <p><a href="#" @click.prevent="serverReset">Server reset&restart</a></p> -->
    </aside>
    <article class="main">
       <!-- <ImagesUploader @changed="handleImages" @response="(msg) => MainContent = msg"/> -->

      <compImgUngrouped v-if="compImgUngroupedToggler"/>

      <compPeopleList v-if="compPeopleListToggler"/>

      <compGroupsShow v-if="compGroupsToggler"/>

      <ImagesVerifiedWithUpload
        v-if="imgUploaderWithUpToggler"
        @response="imgOnSubmit"
      />

      <ImagesVerifiedWithoutUpload
        v-if="imgUploaderWithoutUpToggler"
        @response="imgOnSubmit"
      />

      <ImagesUploaderAdv
        v-if="imgUploaderAdvToggler"
        @changed="handleImages"
        @response="(msg) => MainContent = msg"
      />

      <zipUploader
        v-if="zipUploaderToggler"
        @response="zipOnSubmit"
      />

      <p>{{ MainContent }}</p>
      <p><span v-html="MainContentRaw"></span></p>


    </article>
  </div>
  
</template>


<style>
#app {
  font-family: Avenir, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  /* text-align: center; */
  color: #2c3e50;
  /* margin-top: 20px; */
}

.wrapper {
  display: flex;  
  flex-flow: row wrap;
  /* text-align: center;  */
}

.wrapper > * {
  padding: 0px 20px 20px 20px;
  flex: 1 100%;
}

.header {
  text-align: center;
}

header > h1 {
  background-color: #000;
  color: #fff;
}

.footer {
/*   background: lightgreen; */
}

.main {
  text-align: left;
  padding-right: 0;
/*   background: deepskyblue; */
}

.aside-1 {
  background: #f4f6ff;
}

@media all and (min-width: 600px) {
  .aside { flex: 1 0 0; }
}

@media all and (min-width: 800px) {
  .main    { flex: 3 0px; }
  .aside-1 { order: 1; } 
  .main    { order: 2; }
  .footer  { order: 4; }
}

body {
  padding: 1em; 
}

h3 {
  margin: 40px 0 0;
}

ul {
  list-style-type: none;
  padding: 0;
}

li {
  display: block;
  margin: 0 10px;
}

a {
  color: #42b983;
}


/* Images grid */

.container {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  margin: 0 auto;
}

.row {
  width: 100%;
  height: 100%;
  max-width: 1000px;
  margin: 10px 0;
  display: flex;
  flex-wrap: wrap;
}

.img_thumb {
  max-width: 150px;
}

</style>
