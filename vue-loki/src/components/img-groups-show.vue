<script>
import compGroupItems from './img-groups-items-show.vue'

export default {
    components: {
      compGroupItems
  },

  data() {
    return {
      groups: null,
      all_grouped: false
    }
  },

  methods: {
    async ListGroups() {
      const requestOptions = {
        method: "POST",
      };
      const res = await fetch(`http://127.0.0.1:8000/fr/utility/get_groups`,requestOptions)
      this.groups = await res.json()
      // console.log(this.groups)
      //console.log(this.imgs)
      
    },

    async fetchImg(img) {
        console.log(img)
    },

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
    this.ListGroups()
  },
}
</script>


<template>
<h2>List groups</h2>

<span v-for="group in groups" :key="parseInt(group)">
  <div v-if="group != -1">
      <compGroupItems :group_name="parseInt(group)"></compGroupItems>
  </div>
</span>

</template>


<style scoped>
.flex_container {
  display: flex;
}

.thumb {
    max-width: 80px;
}
</style>