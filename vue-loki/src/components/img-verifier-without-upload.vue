<template>
<h2>Single image verifier (without upload)</h2>
    <p>
        <input
            type="file" 
            @change="onChange($event)"
            accept=".jpg"
        />
    </p>
    <!-- {{ returnData }} -->

    <div v-if="returnToggler">
        <div v-for="item in returnData" :key="item.unique_ids">
            <img @click="clickImg(item)" :src="'/data/'+item.image_names" class="img_thumb"/>
        </div>
    </div>
</template>

<script>
// import { nextTick } from 'vue'
export default {
    emits: ['response', 'change'],

    data() {
        return {
            returnToggler: true,
            returnData: ''
        }
    },

    methods: {
        clickImg(item) {
            console.log(item.unique_ids)
        },
        
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
            console.log('test')
            this.fetchData(evt)
                .then(
                    //data => console.log(data[0].image_names),
                    data => this.returnData = data,
                    // console.log('test2'),
                    // nextTick(() => {
                    //     console.log(this.returnData)
                    // })
                )
                
        },
    },
}
</script>