<script>
import compGroupItems from './img-groups-items-show.vue'

export default {
    components: {
      compGroupItems
  },

  data() {
    return {
      people: null,
      all_grouped: false
    }
  },

  methods: {
    async ListPeople() {
      const requestOptions = {
        method: "POST",
      };
      const res = await fetch(`http://127.0.0.1:8000/fr/people/list`,requestOptions)
      this.people = await res.json()
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
    this.ListPeople()
  },
}
</script>


<template>
<h2>List people</h2>

<span v-for="person in people" :key="person.id">
      <compGroupItems :person_id="person.id" :person_name="person.name"></compGroupItems>
      <!-- {{group.id}}, -->

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