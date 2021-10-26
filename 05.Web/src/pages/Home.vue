<template>
  <q-page class="flex justify-between page-container-class">
    <div class="container-image-input">
      <ImageInput
        class="input-conmponent"
      />
    </div>
    <div class="container-image-prediction">
      <ImagePrediction
        class="element-container1"
        label="Firma 1"
        :url="data.rutafirma1"
        :prediction="data.firma1"
      />
      <ImagePrediction
        class="element-container2"
        label="Firma 2"
        :url="data.rutafirma2"
        :prediction="prediction.firma2"
      />
      <ImagePrediction
        class="element-container3"
        label="Fecha"
        :url="data.rutafecha"
        :prediction="data.fecha"
      />
    </div>
  </q-page>
</template>

<script>

import { mapState, mapActions } from 'vuex';
import ImagePrediction from 'src/components/ImagePrediction.vue'
import ImageInput from 'src/components/ImageInput.vue'

export default {
  name: 'PageIndex',
  components: { 
    ImagePrediction,
    ImageInput
  },
  data () {
    return {
      //  data: {
      //   id: 1,
      //   rutafirma1: "images/C_firma1.jpg",
      //   rutafirma2: "images/C_firma2.jpg",
      //   rutafecha: "images/C_fecha.jpg",
      //   name: "Lorem Ipsum",
      //   firma1: "1",
      //   firma2: "1",
      //   fecha: "29-01-2020"
      // }
      data: {},
      history_predictions_key: "predictions"
    }
  },
  created() {
    this.resetHandlerState(false)
  },
  computed: {
    ...mapState('predictions', [
      'prediction',
      'handler_state'
    ])
  },
  methods: {
    ...mapActions('predictions', [
      'resetHandlerState'
    ]),
  },
  watch: {
    handler_state: {
      handler(newVal) {
        if(newVal === true) {
          this.data = Object.assign({},this.prediction);
          let predictions = this.$q.localStorage.getItem(this.history_predictions_key) ?? []
          const count_id = predictions.length + 1
          const prediction = {
            id: count_id,
            firma1: this.data.firma1,
            firma2: this.data.firma2,
            fecha: this.data.fecha,
            name: this.data.name,
            time: this.data.time
          }
          predictions.push(prediction)
          this.$q.localStorage.set(this.history_predictions_key,predictions)
          this.resetHandlerState(false);
        }
      }
    }
  },
}
</script>

<style lang="scss">
.page-container-class {
 max-width: 1200px;
 margin: 48px auto 0;
}
.container-image-prediction {
  display: grid;
  grid-template-rows: 250px 250px 250px;
  margin: 0 auto;
  gap: 20px 0;
}

.element-container1 {
  grid-row: 1 / 2;
}

.element-container2 {
  grid-row: 2 / 3;
}

.element-container3 {
  grid-row: 3 / 4;
}
</style>