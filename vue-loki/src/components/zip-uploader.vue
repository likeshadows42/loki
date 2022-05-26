<script>
import axios from "axios"
export default {
    emits: ['response'],
    
    data() {
        return {
        MainContentRaw: null
        }
    },

    methods: {
        async uploadZip(evt) {
            const file  = evt.target.files[0]
            let formData = new FormData()
            formData.append('myfile', file)
            formData.append('verifier_names', 'ArcFace')

            this.MainContentRaw = "Data is loaded and analysed.<br>Please wait."
    
            const params = {
                headers: {
                    'Content-Type': 'multipart/form-data'
                } 
            }
            const response = await axios.post(`/api/fr/faces/import_from_zip`, formData, params)
            // return response.json()
            return response.data
            
        },

        onChange(evt) {
            // this.fetchData(evt)
            this.uploadZip(evt)
                .then(data => this.$emit('response', data))
        },
    },
}
</script>

<template>
    <h2>Upload an image archive</h2>
        <input
            type="file" 
            @change="onChange($event)"
            accept=".zip"
        />

    <p><span v-html="MainContentRaw"></span></p>
</template>