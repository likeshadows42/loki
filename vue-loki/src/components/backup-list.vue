<script>
import axios from "axios"

export default {

  data() {
    return {
      backup_list: null,
    }
  },

  methods: {
    async axiosPost(url, params={}) {
      try {
        const res = await axios.post(url, params)
        // console.log(res.data)
        return res.data
      } catch(error) {
        console.log(error)
      }
    },
    
    async getBackups() {
      this.backup_list = await this.axiosPost(`/api/fr/utility/backup/list`, {})
    },

    async restoreZip(name) {
      if(confirm("Are you sure you want tot restore this zip? All the actual content will be replaced.")) {
        // console.log(name)
        await this.axiosPost(`/api/fr/utility/restore_state?file_fp=${encodeURIComponent(name)}`, {})
      }
    },

    checkList(array) {
      return array?.length == 0
    },
  },  

  mounted() {
    this.getBackups()
  },
}
</script>


<template>
<h2>Backup available</h2>

<span v-for="backup in backup_list" :key="backup">
      <div><a @click="restoreZip(backup)" href="#">{{backup}}</a></div>
</span>
<p v-if="checkList(backup_list)">No backup yet.</p>
</template>


<style scoped>
</style>