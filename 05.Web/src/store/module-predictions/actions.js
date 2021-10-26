import { axiosInstance } from "boot/axios";

import { Loading, QSpinnerHourglass } from "quasar";

const spinner =
  typeof QSpinnerHourglass !== "undefined"
    ? QSpinnerHourglass
    : Quasar.components.QSpinnerHourglass;


export function getPredictions({ commit }, payload) {
    return new Promise((resolve, reject) => {
      Loading.show({
        spinner,
        spinnerColor: "light-blue-5",
        messageColor: "grey-3",
        message: "Loading..."
      })
        axiosInstance({
            method: "POST",
            url: "http://localhost:5000",
            header: {
              'Access-Control-Allow-Origin': '*'
            },
            data: {
              id: payload.id,
              sample: payload.image,
              name: payload.name
            }
        })
            .then(response => {
              const data = response.data.data;
              commit('setPredictions', data)
              Loading.hide()
              resolve(data)
            })
            .catch(error => {
              Loading.hide()
              reject(error.response.data.errors)
            })
    })
}

export function resetHandlerState( {commit}, payload ) {
  // console.log(`payload: ${payload}`)
  commit("setHandlerState", payload);
}



export function getS3Url( {commit}, payload) {
  return new Promise((resolve, reject) => {
    axiosInstance({
      method: 'PUT',
      url: payload.url,
      data: payload.file,
      header: {
        "content-type": payload.file.type
      }
    })
        .then(response => {
          resolve(response)
        })
        .catch(error => {
          reject(error.response)
    })
  })
}

export function getSignedUrl() {
  return new Promise((resolve, reject) => {
    axiosInstance({
      method: 'GET',
      baseURL: "https://a7h4yzeuc9.execute-api.us-east-2.amazonaws.com/default/getPresignedUrl"
    })
    .then(response => {
      resolve(response.data.uploadURL)
    })
    .catch(error => {
      reject(error.response)
    })
  })
}
