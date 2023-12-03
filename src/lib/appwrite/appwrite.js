import { Client, Account } from "appwrite";

const client = new Client();

client.setEndpoint("https://cloud.appwrite.io/v1").setProject("nft-mobile-app"); // Replace with your project ID

const urlParam = new URLSearchParams(window.location.search);
const [userId, secret] = [urlParam.get("userId"), urlParam.get("secret")];
const account = new Account(client);
var response = account.updateVerification(userId, secret);
response.then((value) => {
	console.log(value);
	window.location.replace();
});

export { ID } from "appwrite";
