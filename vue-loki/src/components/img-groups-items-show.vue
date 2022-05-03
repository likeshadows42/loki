<script>
import axios from "axios"

export default {

  props: {
    person_id: Number,
    person_name: String
  },

  data() {
    return {
      group_obj: null,
      group_num: 0,
      person_new_name: this.person_name,
      name_title: this.person_name
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

    async getGroupMembers(person_id) {
      const params = {}
      this.group_obj = await this.axiosPost(`http://127.0.0.1:8000/fr/people/get_faces?person_id=${person_id}`, params)
      this.group_num = Object.keys(this.group_obj).length
      //console.log(this.group_obj)
    },

    clickImg(item) {
            console.log(item.unique_id)
    },

    async removeImgFromGroup(person_id, uuid) {
      const uuid_list = new Array(uuid)
      // const params = {uuid_list}
      await this.axiosPost(`http://127.0.0.1:8000/fr/utility/remove_from_group`, uuid_list)
      await this.axiosPost(`http://127.0.0.1:8000/fr/utility/update_record/?term=${uuid}&new_name_tag=`)
      this.getGroupMembers(person_id)
    },

    async changeName(person_id) {
      const params = {}
      await this.axiosPost(`http://127.0.0.1:8000/fr/people/set_name?person_id=${person_id}&person_name=${this.person_new_name}`, params)
      this.name_title = this.person_new_name
      this.getGroupMembers(this.person_id)
      // alert(this.person_new_name+", "+person_id)

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
    this.getGroupMembers(this.person_id)
  },
}
</script>


<template>

  <div class="group_div">
    <div class="header_div">
      <h3 v-if="this.name_title">#{{ person_id }} {{this.name_title}}</h3>
      <h3 v-else>#{{ person_id }} (unamed)</h3>
      <div><input v-model="person_new_name"> <button @click="changeName(person_id)">CHANGE</button></div>
      <div>Num pics: {{group_num }}</div>
    </div>
    <div class="imgs_container">
      <div v-for="item in group_obj" :key="item.id" class="img_group">
        <div class="img_div">
          <!-- <img @click="removeImgFromGroup(person_id, item.id)" :src="'/data/'+item.image_name" class="img_thumb"/> -->
          <img :src="'/data/'+item.image_name_orig" class="img_thumb"/>
        </div>
        <!-- <div v-if="item.name_tag">{{item.name_tag}}</div><div v-else>no tag</div> -->
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