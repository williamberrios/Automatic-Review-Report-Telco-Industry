<template>
  <div 
    v-if="predictions.length"
    style="margin: 0 auto;"
  >
    <div class="title-container">
      Historial de Predicciones
    </div>  
    <q-card class="card-container">
      <q-card-section
        class="header-row"
      >
        <div class="header-row-id">
          ID
        </div>
        <div class="header-row-name">
          Nombre
        </div>
        <div class="header-row-prediction-firma1">
          Firma 1
        </div>
        <div class="header-row-prediction-firma2">
          Firma 2
        </div>
        <div class="header-row-prediction-fecha">
          Fecha Extraida
        </div>
        <div class="header-row-prediction-hora">
          Hora Subida
        </div>
      </q-card-section>
      <q-card-section
        v-for="(prediction,index) in predictions"
        :key="prediction.id"
        :index="index"
        class="section-row"
      >
        <div class="section-row-id">
          {{prediction.id}}
        </div>
        <div class="section-row-name">
          {{prediction.name}}
        </div>
        <div class="section-row-prediction-firma1">
          {{prediction.firma1}}
        </div>
        <div class="section-row-prediction-firma2">
          {{prediction.firma2}}
        </div>
        <div class="section-row-prediction-fecha">
          {{prediction.fecha}}
        </div>
        <div class="section-row-prediction-hora">
          {{prediction.time}}
        </div>
      </q-card-section>
    </q-card>
  </div>
</template>

<script>

export default {
  name: 'History',
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
      //   fecha: "29-01-2020",
      //   time: "15:30"
      // }
      predictions: [],
      history_predictions_key: "predictions"
    }
  },
  created() {
    this.predictions = this.$q.localStorage.getItem(this.history_predictions_key)
  }
}
</script>

<style lang="scss">
  .title-container {
    margin: 48px auto;
    width: 100%;
    font-size: 20px;
    font-weight: 600;
    text-align: center;
  }
  .card-container {
    max-width: 1200px;
    margin: 0 auto;
  }
  .header-row {
    display: grid;
    grid-template-columns: 90px 1fr 1fr 1fr 1fr 1fr;
    margin: 0 auto;
    font-weight: 600;
    font-size: 18px;
    &-id {
      grid-column: 1 / 2;
      justify-self: center;
    }
    &-name {
      grid-column: 2 / 3;
      justify-self: center;
    }
    &-prediction-firma1 {
      grid-column: 3 / 4;
      justify-self: center;
    }
    &-prediction-firma2 {
      grid-column: 4 / 5;
      justify-self: center;
    }
    &-prediction-fecha {
      grid-column: 5 / 6;
      justify-self: center;
    }
    &-prediction-hora {
      grid-column: 6 / 7;
      justify-self: center;
    }
  }
  .section-row {
    display: grid;
    grid-template-columns: 90px 1fr 1fr 1fr 1fr 1fr;
    margin: 0 auto;
    font-size: 16px;
    &-id {
      grid-column: 1 / 2;
      justify-self: center;
    }
    &-name  {
      grid-column: 2 / 3;
      justify-self: center;
    }
    &-prediction-firma1 {
      grid-column: 3 / 4;
      justify-self: center;
    }
    &-prediction-firma2 {
      grid-column: 4 / 5;
      justify-self: center;
    }
    &-prediction-fecha {
      grid-column: 5 / 6;
      justify-self: center;
    }
    &-prediction-hora {
      grid-column: 6 / 7;
      justify-self: center;
    }
  }
</style>