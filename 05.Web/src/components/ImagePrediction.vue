<template>
  <div class="container-element">
    <div class="container-element-label">
      {{label}}
    </div>
    <div class="container-element-prediction-image">
      <q-img
        :src="url"
        spinner-color="white"
        style="width: 300px;"
      />
    </div>
    <div class="container-element-prediction-text">
      <span class="container-element-prediction-text-label">Prediction:</span> 
      <span class="container-element-prediction-text-value">{{prediction}}</span>
    </div>
  </div>
</template>

<script>
export default {
  props: { 
    label: {
      type: String,
      required: true
    },
    url: {
      type: String,
      default: "images/image_default.jpg"
    },
    prediction: {
      default: "-"
    }
  },
  watch: {
    url: {
      handler(newVal) {
        console.log(newVal);
      }
    }
  },
  name: 'ImagePrediction',
  data () {
    return {
      // essentialLinks: linksData
    }
  }
}
</script>

<style lang="scss">
.container-element {
  display: grid;
  grid-template-rows: auto auto auto;
  gap: 8px 0;
  &-label {
    grid-row: 1 / 2;
    text-align: center;
    font-size: 20px;
  }
  &-prediction-image {
    grid-row: 2 / 3;
    .q-img__image {
      border-radius: 5px;
    }
  }
  &-prediction-text {
    grid-row: 3 / 4;
    font-size: 18px;
    display: flex;
    flex-flow: row wrap;
    justify-content: space-between;
    &-label {
      width: 40%;
      text-align: center;
    }
    &-value {
      width: 50%;
      text-align: center;
    }
  }
}
</style>
