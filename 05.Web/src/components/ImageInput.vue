<template>
<div class="image-input-container">
  <div class="image-input-container-image">
    <q-img
      :src="url ? url : 'images/image_default.jpg'"
      spinner-color="white"
      style="width:400px;cursor: pointer;border-radius: 5px;"
      :ratio="1"
      @click="$refs.fileInput.click()"
    />    
    <input
      @change="onFileChange"
      label="Pick one file"
      ref="fileInput"
      type="file"
      accept="image/*"
      style="max-width: 300px;display: none"
    />
  </div>
  <div class="image-input-container-btns">
    <div class="upload-btn">
      <q-btn 
        push 
        no-caps
        color="primary"  
        label="Upload Image" 
        @click="$refs.fileInput.click()"
      />
    </div>
    <div class="prediction-btn">
      <q-btn 
        push 
        no-caps
        color="primary" 
        label="Get Predictions" 
        @click="GetPredictions()"
      />
    </div>
  </div>
</div>
</template>

<script>

import { mapState, mapActions } from 'vuex';
// import aws from 'aws-sdk';
// import crypto from 'crypto';
// import { promisify } from 'util';
// const randomBytes = promisify(crypto.randomBytes);

// const region = process.env.region;
// const bucketName = process.env.bucketName;
// const accessKeyId = process.env.accessKeyId;
// const secretAccessKey = process.env.secretAccessKey;


// const s3 = new aws.S3({
//   region,
//   accessKeyId,
//   secretAccessKey
// })

const MAX_IMAGE_SIZE = 1000000

export default {
  name: 'ImageInput',
  data () {
    return {
      url: "",
      image: '',
      file_image: '',
      name: ''
    }
  },
  computed: {
    ...mapState('predictions', [
      'prediction'
    ]),
    // baseURL() {
    //   return 'https://amplify-amplifygraphs-dev-01149-deployment.s3.us-east-2.amazonaws.com'
    // },
    // S3Client() {
    //   return new aws.S3(this.config)
    // },
    // newFileName() {
    //   return Math.random().toString().slice(2)
    // }
  },
  methods: {
    ...mapActions('predictions', [
      'getPredictions',
      'getS3Url',
      'getSignedUrl',
    ]),
    createImage (file) {
      // var image = new Image()
      let reader = new FileReader()
      reader.onload = (e) => {
        console.log('length: ', e.target.result.includes('data:image/jpeg'))
        if (!e.target.result.includes('data:image/jpeg')) {
          return alert('Wrong file type - JPG only.')
        }
        if (e.target.result.length > MAX_IMAGE_SIZE) {
          return alert('Image is loo large.')
        }
        this.image = e.target.result
      }
      reader.readAsDataURL(file)
      this.uploadImage();
    },
    onFileChange (e) {
      let files = e.target.files || e.dataTransfer.files
      if (!files.length) return
      this.file_image = files[0]
      const index_format = this.file_image.name.lastIndexOf('.')
      this.name = this.file_image.name.slice(0,index_format)
      this.createImage(files[0])
    },
    async uploadImage() {
      // console.log('Upload clicked')
      // Get the presigned URL
      const presignedUrl = await this.getSignedUrl()
      // console.log('Response: ', presignedUrl)
      // console.log('Uploading: ', this.image)
      // let binary = atob(this.image.split(',')[1])
      // let array = []
      // for (var i = 0; i < binary.length; i++) {
      //   array.push(binary.charCodeAt(i))
      // }
      // let blobData = new Blob([new Uint8Array(array)], {type: 'image/jpeg'})
      // console.log(blobData)
      // console.log('Uploading to: ', presignedUrl)
      // const result = await fetch(presignedUrl, {
      //   method: 'PUT',
      //   body: blobData
      // })
      const payload = {
        url: presignedUrl,
        file: this.file_image
      }
      // console.log(payload)
      await this.getS3Url(payload)
      // console.log('Result: ', result)
      // Final URL for the user doesn't need the query string params
      this.uploadURL = presignedUrl.split('?')[0]
      this.url = this.uploadURL
      // console.log(this.uploadURL)
    },
    GetPredictions() {
      const payload = {
        id: 1,
        image: this.url,
        name: this.name
      }
      this.getPredictions(payload)
    }
  }
}
</script>

<style lang="scss">
.image-input-container {
  display: flex;
  flex-direction: column;
  &-image {
    display: row;
    height: 400px;
  }
  &-btns {
    display: flex;
    flex-flow: row wrap;
    justify-content: space-between;
    margin-top: 16px;
  }
}
</style>