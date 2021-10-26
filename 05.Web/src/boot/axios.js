import Vue from 'vue'
import axios from "axios"

const axiosInstance = axios.create()

Vue.prototype.$axios = Vue.axios = axiosInstance

export { axiosInstance }

