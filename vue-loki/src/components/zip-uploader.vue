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

        displayResult(msg){
            this.MainContentRaw += "<p><b>All done!</b></p>"
            this.MainContentRaw += '<p>' + msg.message + '</p>'
            this.MainContentRaw += '<p>New images: <b>' + msg.n_records + '</b></p>'
            this.MainContentRaw += '<p>Skipped files: <b>' + msg.n_skipped + '</b></p>'
            this.$emit('response', msg)
        },

        onChange(evt) {
            // this.fetchData(evt)
            this.uploadZip(evt)
                // .then(data => this.$emit('response', data))
                .then(data => this.displayResult(data))
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