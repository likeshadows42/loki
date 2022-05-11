<script>
import compGroupItemsCanvas from './img-groups-item-canvas.vue'

export default {
  components: {
      compGroupItemsCanvas
  },

  data() {
    return {
      imgs: null,
      item_id_selected: null,
      people_list: [],
      people_list_hidden: null,
      people_list_selected: null,
      people_lookup_table: null
    }
  },

  methods: {
    async fetchUngrouped() {
      const requestOptions = {
        method: "POST",
      };
      const res = await fetch(`http://127.0.0.1:8000/fr/facerep/get_ungrouped`,requestOptions)
      this.imgs = await res.json()
    },

    async ListPeople() {
      const requestOptions = {
        method: "POST",
      };
      const res = await fetch(`http://127.0.0.1:8000/fr/people/list`,requestOptions)
      const json = await res.json()
      return json      
    },

    fetchImg(item_id) {        
        this.item_id_selected = item_id
    },
    
    async populatedAutocomplete() {
      const response = await this.ListPeople()
      // console.log(response)
      const names_list  = []
      const names_lookup = []
        for(const item of response) {
          if(item.name != null) {
            names_lookup.push(  {
                                text: item.name,
                                value: item.id
                              }
                            )
            names_list.push(item.name)
          }
        }
      this.people_list = [...names_list]
      this.people_lookup_table = [...names_lookup]
    },

    displayCombobox(id) {
      if(id == this.item_id_selected) {
        return true
      }
      return false
    },

    async setPerson(ev) {
      var person_value
      if(ev.key == 'Enter') {
        // console.log(this.people_lookup_table['[Target]'])
        if(this.people_list_hidden != null) {
          const person = this.people_lookup_table.find(arr => arr.text == this.people_list_hidden)
          if(!person) {
            // Create a new person
            const requestOptions = {
              method: "POST",
            }
            const res = await fetch(`http://127.0.0.1:8000/fr/people/add_new?person_name=${this.people_list_hidden}`,requestOptions)
            const result = await res.json()
            console.log(result.id)
            person_value = result.id
          } else {
            person_value = person.value
          }

          const requestOptions = {
            method: "POST",
          }
          const res = await fetch(`http://127.0.0.1:8000/fr/people/assign_facerep?person_id=${person_value}&facerep_id=${this.item_id_selected}`,requestOptions)
          const json = await res.json()
          if(json != 'ok') {
            throw 'Error!'
          }
          this.fetchUngrouped()
        }
      }
    },

    showEv() {
      // console.log(ev)
    },

    checkImgs(array) {
      return array?.length == 0
    }

  },  

  created() {
  },

  mounted() {
    this.fetchUngrouped()
    this.populatedAutocomplete()
  },
}
</script>


<template>
<h2>List all ungrouped images</h2>

<!-- <div v-if="item_id_selected">{{ this.item_id_selected }}</div>
<div v-if="people_list_selected">{{ this.people_list_selected}} </div> -->

<!-- <vue3-simple-typeahead
  id="typeahead_1"
  placeholder = "Associate this image to an existing person or type a new one"
  :items = "people_list"
  :minInputLength="2"
  @selectItem="selectPerson"
>
</vue3-simple-typeahead> -->

<span v-for="img in imgs" :key="img.id">
    <compGroupItemsCanvas :item="img" @parent-handler="fetchImg"></compGroupItemsCanvas>
    <span v-if="displayCombobox(img.id)">
      <v-combobox
        v-model="people_list_hidden"
        :items="people_list"
        no-data-text="No people in the database"
        placeholder="Input the name of the person (existing or not)"
        persistent-placeholder
        @input="showEv"
        @keydown="setPerson"
      >
      </v-combobox>
    </span>
</span>
<p v-if="checkImgs(imgs)">No ungrouped images</p>

</template>


<style scoped>
.thumb {
    max-width: 80px;
}
</style>