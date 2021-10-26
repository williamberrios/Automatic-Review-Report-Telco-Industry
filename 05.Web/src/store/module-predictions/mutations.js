export function setPredictions(state, value){
  state.prediction = value
  state.handler_state = true
}

export function setHandlerState(state, value) {
  state.handler_state = value
}
