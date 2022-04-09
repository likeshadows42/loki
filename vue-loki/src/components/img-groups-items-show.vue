<script>
import axios from "axios"

export default {

  props: {
    group_name: Number
  },

  data() {
    return {
      group_obj: null,
      // all_grouped: false
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

    async getGroupMembers(group) {
      console.log(`Getting global parameters ${group}`)
      const params = {target_group_no: "0",}
      this.group_obj = await this.axiosPost(`http://127.0.0.1:8000/fr/utility/view_by_group_no`, params)
      console.log(this.group_obj)
    },

    // async fetchImg(img) {
    //     console.log(img)
    // },

    // checkGroup(num) {
    //   if(num == -1) {
    //     return true
    //   } else {
    //     this.all_grouped = true
    //     return false
    //   }
    // }

        // removeImg(img) {
    //   this.imgs = this.imgs.filter((t) => t !== img)
    // }, 
  },
  mounted() {
    this.getGroupMembers(this.group_name)
    // console.log(this.group_obj)
  },
}
</script>


<template>
<div>{{ group_name }}</div>

<!-- <span v-for="item in group_obj" :key="group">
  <span v-if="group != -1">
    <div>{{ group }}</div>
  </span>
</span> -->

<!-- <p><button @click="getList()">get list elements</button></p> -->

<!-- <span v-for="img in imgs" :key="img.unique_id">
  <span v-if="checkGroup(img.group_no)">
    <img @click="fetchImg(img.image_name)" :src="`/data/${img.image_name}`" class="thumb">
  </span>
</span>
<p v-if="all_grouped == true">No ungrouped images</p> -->

</template>


<style scoped>
.thumb {
    max-width: 80px;
}
</style>