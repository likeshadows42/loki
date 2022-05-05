<script>
import axios from "axios"
import compGroupItemsCanvas from './img-groups-item-canvas.vue'

export default {
  components: {
      compGroupItemsCanvas
  },

  props: {
    person_id: Number,
    person_name: String
  },

  data() {
    return {
      group_obj: null,
      group_num: 0,
      person_new_name: this.person_name != null ? this.person_name : '',
      person_name_placeholder: this.person_name != null ? this.person_name : 'unamed'
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
      // return this.group_obj
      //console.log(this.group_obj)
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

    // removeImg(img) {
    //   this.imgs = this.imgs.filter((t) => t !== img)
    // },

  },

  created() {
    this.getGroupMembers(this.person_id)
  },

  mounted() {
    // this.getGroupMembers(this.person_id)
  },
}
</script>


<template>

  <div class="group_div">
    <div class="header_div">
      <!-- <div>#{{ person_id }} <input v-model="person_new_name" class="person_name_box" :size="person_new_name.length"> <button @click="changeName(person_id)">CHANGE</button></div> -->
      <div>#{{ person_id }} <input v-model="person_new_name" :placeholder="person_name_placeholder" class="person_name_box" :size="person_new_name.length != 0 ? person_new_name.length: 5"> <button @click="changeName(person_id)">CHANGE</button></div>
      <div>Num pics: {{group_num }}</div>
    </div>
    <div class="imgs_container">
      <div v-for="item in group_obj" :key="item.id" class="img_group">
        <div class="img_div">
          <compGroupItemsCanvas :item="item"></compGroupItemsCanvas>
        </div>
      </div>
    </div>
  </div>
  <p v-if="group_num == 0">No images</p>

</template>


<style scoped>
.group_div {
  padding-bottom: 50px;
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

.person_name_box {
  border: none;
  font-size: 1.17em;
  font-weight: bold;
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