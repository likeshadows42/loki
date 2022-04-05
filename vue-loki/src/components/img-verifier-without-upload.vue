<template>
<h2>Single image verifier (without upload)</h2>
    <p>
        <input
            type="file" 
            @change="onChange($event)"
            accept=".jpg"
        />
    </p>  
</template>

<script>
export default {
    emits: ['response'],

    methods: {
        async fetchData(evt) {
            let formData = new FormData()
            formData.append('files', evt.target.files[0])

            const stringAPI = 'fr/verify/no_upload?detector_name=retinaface&verifier_name=ArcFace&align=true&normalization=base&metric=cosine&threshold=-1&verbose=false'
            
            const fetchResult = await fetch(`http://localhost:8000/`+stringAPI,
                {
                    method: 'POST',
                    body: formData,
                }
            )
            return fetchResult.json()

            // const fetchResult = await fetch(`http://localhost:8000/`+stringAPI,
            //                                 {
            //                                     method: 'POST',
            //                                     body: formData,
            //                                 }
            //                                 )
            // const data = await fetchResult.json()
            
            // if (fetchResult.ok) {
            //     console.log(`Fecthed news data successfully`)
            //     console.log(data)
            //     return data
            // } else {
            //     console.log(data.message, data.data, data.code)
            // }
        },

        onChange(evt) {
            this.fetchData(evt)
                .then(data => this.$emit('response', data))
        },
    },
}
</script>