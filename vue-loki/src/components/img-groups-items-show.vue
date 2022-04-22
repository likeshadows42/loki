<script>
import axios from "axios"

export default {

  props: {
    group_name: Number
  },

  data() {
    return {
      group_obj: null,
      group_num: 0,
      person_name: '',
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
        return await res.data
      } catch(error) {
        console.log(error)
      }
    },

    async getGroupMembers(group) {
      const params = {}
      this.group_obj = await this.axiosPost(`http://127.0.0.1:8000/fr/utility/view_by_group_no?target_group_no=${group}`, params)
      this.group_num = Object.keys(this.group_obj).length
      //console.log(this.group_obj)
    },

    clickImg(item) {
            console.log(item.unique_id)
    },

    async removeImgFromGroup(group_name, uuid) {
      const uuid_list = new Array(uuid)
      // const params = {uuid_list}
      await this.axiosPost(`http://127.0.0.1:8000/fr/utility/remove_from_group`, uuid_list)
      await this.axiosPost(`http://127.0.0.1:8000/fr/utility/update_record/?term=${uuid}&new_name_tag=`)
      this.getGroupMembers(group_name)
    },

    async changeTag(group_name) {
      const params = {}
      await this.axiosPost(`http://127.0.0.1:8000/fr/utility/edit_tag_by_group_no?target_group_no=${group_name}&new_name_tag=${this.person_name}`, params)
      this.getGroupMembers(this.group_name)
    },

    say() {
      alert(this.person_name)
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
  },
}
</script>


<template>

  <div class="group_div">
    <div class="header_div">
      <h3>Group #{{ group_name }}</h3>
      <div>Name this person: <input v-model="person_name"> <button @click="changeTag(group_name)">CHANGE</button></div>
      <div>Num pics: {{group_num }}</div>
    </div>
    <div class="imgs_container">
      <div v-for="item in group_obj" :key="item.unique_id" class="img_group">
        <div class="img_div">
          <img @click="removeImgFromGroup(group_name, item.unique_id)" :src="'/data/'+item.image_name" class="img_thumb"/>
        </div>
        <div v-if="item.name_tag">{{item.name_tag}}</div><div v-else>no tag</div>
      </div>
    </div>
  </div>


  <p v-if="group_num == 0">No ungrouped images</p>

</template>


<style scoped>
.group_div {
}

.header_div {
  /* background-color: #f4f6ff; */
}

.imgs_container {
  display: flex;
  flex-wrap: wrap;
}

.img_group {
  
}

.img_div {
  padding-right: 10px;
  /* min-height: 100%; */
}

.img_thum {
    max-width: 80px;
    /* height: 100%; */
}
</style>