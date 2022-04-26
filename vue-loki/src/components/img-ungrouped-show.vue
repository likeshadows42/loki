<script>
export default {
  
  data() {
    return {
      imgs: null,
      all_grouped: false
    }
  },

  methods: {
    async fetchData() {
      const requestOptions = {
        method: "POST",
      };
      const res = await fetch(`http://127.0.0.1:8000/fr/utility/view_database?amt_detail=complete&output_type=structure`,requestOptions)
      this.imgs = await res.json()
      // console.log(`Fecthed news data successfully`)
      //console.log(this.imgs)
      
    },

    async fetchImg(img) {
        console.log(img)
    },

    checkGroup(num) {
      if(num == -1) {
        return true
      } else {
        this.all_grouped = true
        return false
      }
    }

        // removeImg(img) {
    //   this.imgs = this.imgs.filter((t) => t !== img)
    // },
  },  

  mounted() {
    this.fetchData()
  },
}
</script>


<template>
<h2>List all untagged images</h2>

<!-- <p><button @click="getList()">get list elements</button></p> -->

<span v-for="img in imgs" :key="img.unique_id">
  <span v-if="checkGroup(img.group_no)">
    <img @click="fetchImg(img.image_name)" :src="`/data/${img.image_name}`" class="thumb">
    <!-- <button @click="removeImg(img)">X</button> -->
  </span>
</span>
<p v-if="imgs == true">No ungrouped images</p>

</template>


<style scoped>
.thumb {
    max-width: 80px;
}
</style>