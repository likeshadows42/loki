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
      const res = await fetch(`/api/people/list`,requestOptions)
      this.people = await res.json()
      // console.log(this.groups)
      //console.log(this.imgs)
      
    },

    async fetchImg(img) {
        console.log(img)
    },

    checkPeople(array) {
      return array?.length == 0
    },
  },  

  mounted() {
    this.ListPeople()
  },
}
</script>


<template>
<h2>List people</h2>

<span v-for="person in people" :key="person.id">
      <compGroupItems :person_id="person.id" :person_name="person.name" :person_note="person.note"></compGroupItems>
</span>
<p v-if="checkPeople(people)">No people yet.</p>
</template>


<style scoped>
.flex_container {
  display: flex;
}

.thumb {
    max-width: 80px;
}
</style>