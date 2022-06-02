<script>
import axios from "axios"
import compGroupItemsCanvas from './img-groups-item-canvas.vue'

export default {
  components: {
      compGroupItemsCanvas
  },

  props: {
    person_id: Number,
    person_name: String,
    person_note: String
  },

  data() {
    return {
      group_obj: null,
      group_num: 0,
      person_new_name: this.person_name != null ? this.person_name : '',
      person_name_placeholder: this.person_name != null ? this.person_name : 'unamed',
      person_new_note: this.person_note != null ? this.person_note : '',
      person_note_placeholder: this.person_new_note == null ? 'insert a note' : ''
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
      this.group_obj = await this.axiosPost(`/api/fr/people/get_faces?person_id=${person_id}`, params)
      this.group_num = Object.keys(this.group_obj).length
    },

    async updatePerson(person_id) {
      await this.axiosPost(`/api/fr/people/set_name?person_id=${person_id}&person_name=${this.person_new_name}`, {})
      await this.axiosPost(`/api/fr/people/set_note?person_id=${person_id}&new_note=${this.person_new_note}`, {})
      this.name_title = this.person_new_name
      this.getGroupMembers(this.person_id)
    },

    async removeFace(action, item_id, person_id) {
      const params = {}
      if(action == 'remove') {
        await this.axiosPost(`/api/fr/facerep/unjoin?face_id=${item_id}`, params)
      }
      if(action == "hide") {
        this.hideFace(item_id)
      }
       this.getGroupMembers(person_id)

    },

    async hideFace(item_id) {
       if(confirm("Are you sure this face ? "+item_id)) {
          console.log(item_id)
          const response = await this.axiosPost(`/api/fr/facerep/hide?${item_id}`, {})
          console.log(response)
          this.getGroupMembers(this.person_id)
       }
    },

    async hidePerson(person_id) {
      if(confirm("Are you sure to delete person #"+person_id+"?")) {
        const response = await this.axiosPost(`/api/fr/people/hide?person_id=${person_id}`, {})
        console.log(response)
        this.getGroupMembers(this.person_id)
      }
    },

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
  <div v-if="group_num > 0" class="group_div">
    <div class="header_div">
      <div>
          #{{ person_id }}
          <input v-model="person_new_name" :placeholder="person_name_placeholder" class="person_name_box" :size="person_new_name.length != 0 ? person_new_name.length: 7">
          <button @click="updatePerson(person_id)">SET NAME</button>
          &nbsp;
          <button @click="hidePerson(person_id)">HIDE</button>
      </div>
      Note <textarea :placeholder="this.person_note_placeholder" v-model="person_new_note"></textarea>
      <div>Num pics: {{group_num }}</div>
    </div>
    <div class="imgs_container">
      <div v-for="item in group_obj" :key="item.id" class="img_group">
        <div class="img_div">
          <compGroupItemsCanvas :item="item" :show_button="true" @parent-handler="removeFace"></compGroupItemsCanvas>
        </div>
      </div>
    </div>
  </div>
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

textarea {
  width: 100%;
  resize: none;
  border-color: #ddd;
}
</style>