<template>
  <div class="wrapper">
    <header class="header"><Home msg="Loki MVP"/></header>
    <aside class="aside aside-1">
      <p><a href="#" @click.prevent="getGlobaldata">Show global variables</a></p>
      <p><a href="#" @click.prevent="zipUploaderToggle">Upload zip file</a></p>
      <p><a href="#" @click.prevent="imgUploaderAdvToggle">Upload multiple images</a></p>
      <p><a href="#" @click.prevent="imgVerWithoutUpToggle">Verify image (without upload)</a></p>
      <p><a href="#" @click.prevent="imgVerWithUpToggle">Verify image (with upload)</a></p>
      <!-- <SecondComp @response="(msg) => MainContent = msg"/>
      <SecondComp @response="mainClear"/> -->
      
    </aside>
    <article class="main">
       <!-- <ImagesUploader @changed="handleImages" @response="(msg) => MainContent = msg"/> -->
       
      <GlobalsGet
        v-if="globalDataToggler"
      /> 

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


<script>
// imports
import Home from './components/home.vue'
import GlobalsGet from './components/globals-get.vue'
// import SecondComp from './components/second-comp.vue'
import ImagesVerifiedWithUpload from './components/img-verifier-with-upload.vue'
import ImagesVerifiedWithoutUpload from './components/img-verifier-without-upload.vue'
import ImagesUploaderAdv from './components/img-uploader-adv.vue'
import zipUploader from './components/zip-uploader.vue'

// exports
export default {
  name: 'App',

  components: {
    Home,
    GlobalsGet,
    // SecondComp,
    ImagesVerifiedWithUpload,
    ImagesVerifiedWithoutUpload,
    ImagesUploaderAdv,
    zipUploader
  },

  methods: {
    mainClear() {
      this.imgUploaderWithUpToggler = false
      this.imgUploaderWithoutUpToggler = false
      this.imgUploaderAdvToggler = false
      this.zipUploaderToggler = false
      this.globalDataToggler = false
      this.MainContentRaw = null
    },

    handleImages(files) {
      console.log(files)
    },
    
    zipUploaderToggle() {
      this.MainContent = null
      this.imgUploaderWithUpToggler = false
      this.imgUploaderWithoutUpToggler = false
      this.imgUploaderAdvToggler = false
      this.zipUploaderToggler = true
      this.globalDataToggler = false
      this.MainContentRaw = null
    },

    imgVerWithUpToggle() {
      this.MainContent = null
      this.imgUploaderWithUpToggler = true
      this.imgUploaderWithoutUpToggler = false
      this.imgUploaderAdvToggler = false
      this.zipUploaderToggler = false
      this.globalDataToggler = false
      this.MainContentRaw = null
    },

    imgVerWithoutUpToggle() {
      this.MainContent = null
      this.imgUploaderWithUpToggler = false
      this.imgUploaderWithoutUpToggler = true
      this.imgUploaderAdvToggler = false
      this.zipUploaderToggler = false
      this.globalDataToggler = false
      this.MainContentRaw = null
    },

    imgUploaderAdvToggle() {
      this.MainContent = null
      this.imgUploaderWithUpToggler = false
      this.imgUploaderWithoutUpToggler = false
      this.imgUploaderAdvToggler = true
      this.zipUploaderToggler = false
      this.globalDataToggler = false
      this.MainContentRaw = null
    },

    getGlobaldata() {
      this.MainContent = null
      this.imgUploaderWithUpToggler = false
      this.imgUploaderWithoutUpToggler = false
      this.imgUploaderAdvToggler = false
      this.zipUploaderToggler = false
      this.globalDataToggler = true
      this.MainContentRaw = null
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
        imgs += '<img src="'+ imgBase + imgURL +'" class="img_thumb"/>'
        // imgs += '<p>'+imgURL+'</p>'
      }
      
      // let output = '<div class="container"><div class="row">' + imgs + '</div></div>'

      this.MainContentRaw = imgs
    },

    zipOnSubmit(msg) {
       console.log("Zip received!")
       this.MainContent = msg
    },

  },

  data(){
    return {
      MainContent: null,
      MainContentRaw: null,
      imgUploaderWithUpToggler: false,
      imgUploaderWithoutUpToggler: false,
      imgUploaderAdvToggler: false,
      zipUploaderToggler: false,
      globalDataToggler: false,
    }
  },

  created() {
    // this.const_test1 = 'ok';
  },

  mounted() {
    console.log(`Main app initiated.`)
    // console.log(`Constant "const_test1"= "${this.const_test1}"`)
  },
}
</script>


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
