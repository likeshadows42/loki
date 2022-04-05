<template>
<h2>Upload an image archive</h2>
    <input
        type="file" 
        @change="onChange($event)"
        accept=".zip"
    />
</template>

<script>
export default {
    emits: ['response'],
    methods: {
        async fetchData(evt) {
            let formData = new FormData()
            formData.append('files', evt.target.files[0])

            const stringAPI = 'fr/create_database/from_directory'
            
            const fetchResult = await fetch(`http://localhost:8000/`+stringAPI,
                {
                    method: 'POST',
                    body: formData,
                }
            )
            return fetchResult.json()
        },

        onChange(evt) {
            this.fetchData(evt)
                .then(data => this.$emit('response', data))
        },
    },
}
</script>