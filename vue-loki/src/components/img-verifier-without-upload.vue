<script>
import axios from "axios"
export default {
    emits: ['response', 'change'],

    data() {
        return {
            returnData: null,
            person_name: null,
            alert_text: null,
            image_loaded: null,
        }
    },

    methods: {
        clickImg(item) {
            console.log(item.unique_ids)
        },

        async checkImage(file) {
            const apiURL = '/api/fr/verify/no_upload?threshold=.8'
            let formData = new FormData()
             formData.append('files', file)

            try {
                const res = await axios.post(apiURL, formData, {})
                return res.data
            } catch(error) {
                console.log(error)
            }
        },
 
        async getPersonByID(id) {
            const apiURL = `/api/fr/facerep/get_person?facerep_id=${id}`
            try {
                const res = await axios.post(apiURL)
                return res.data
            } catch(error) {
                console.log(error)
            }

        },

        async onChange(evt) {
            this.alert_text = null
            this.returnData = null 
            this.image_loaded = URL.createObjectURL(evt.target.files[0])
            const res = await this.checkImage(evt.target.files[0])
            if(res[0][0].length > 0) {
                const facerep_id_0 = res[0][0][0].unique_id
                const person = await this.getPersonByID(facerep_id_0)
                if(person.length > 0) {
                    this.person_name = person[0].Person.name
                    this.returnData = res[0][0]
                    return
                }
            }
            this.alert_text = "No match for the image"
        },

        checkThreshold(threshold) {
            if(threshold < .5) {
                return "Equal to"
            }
            else if (threshold >= .5 && threshold < .7) {
                return "In doubt if equal to"
            }
            return "Different from everyone known."
        },

        visibilityToSimilar(threshold) {
            if(threshold < .7) return true
            return false
        }
    },
}
</script>


<template>
<h2>Single image verifier (no upload)</h2>
    <p>
        <input
            type="file" 
            @change="onChange($event)"
            accept=".jpg"
        />
    </p>

    <div v-if="image_loaded">
        <h3>Image loaded</h3>
        <img :src="image_loaded" class="img_thumb">
    </div>

    <div v-if="alert_text">
        <div>&nbsp;</div>
        <div><b>{{ alert_text }}</b></div>
    </div>
    <div v-if="returnData">
        <div v-if="visibilityToSimilar(returnData[0].distance)">
            <h3>Best match</h3>
                <div>{{ this.checkThreshold(returnData[0].distance)}} <b>{{ person_name }}</b></div>
                <div>ID: {{returnData[0].unique_id}}, {{ returnData[0].image_name}}, threshold:{{ returnData[0].distance}})</div>
                <div>&nbsp;</div>
        </div>
        <div v-else>
            <div>&nbsp;</div>
            <div>{{ this.checkThreshold(returnData[0].distance)}}</div>
            <div>&nbsp;</div>
            <div v-if="returnData.length > 0">
                <div>closest match: ID: {{returnData[0].unique_id}}, {{ returnData[0].image_name}}, threshold:{{ returnData[0].distance}}</div>
                <img :src="'/data/'+returnData[0].image_name" class="img_thumb"/>
            </div>
        </div>

        <div v-if="visibilityToSimilar(returnData[0].distance)">
            <img :src="'/data/'+returnData[0].image_name" class="img_thumb"/>

            <div v-if="returnData.length > 1">
                <h3>Other similar faces</h3>
                <div v-for="person in returnData.slice(1)" :key="person.unique_id">
                    <div v-if="visibilityToSimilar(person.distance)">
                        <div><img :src="'/data/'+person.image_name" class="img_thumb"/></div>
                        <div>ID: {{person.unique_id}}, {{ person.image_name}}, threshold:{{ person.distance}}</div>
                        <div>&nbsp;</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>