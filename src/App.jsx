// import React, { useState } from 'react';
// import { account, ID } from './lib/appwrite';
import { Client, Account, ID } from "appwrite";

const App = () => {
	const client = new Client().setEndpoint("https://cloud.appwrite.io/v1").setProject("nft-mobile-app");

	async function emailVerify() {
		var account = new Account(client);
		var urlParams = new URLSearchParams(window.location.search);
		var secret = urlParams.get("secret");
		var userId = urlParams.get("userId");
    await account.createEmailSession('alifnasrulloh.jbg@gmail.com', 'qwertyuiop');
		var promise = account.createVerification("http://localhost:1234/auth/verification");
	}

	return (
		<button
			className="btn btn-primary"
			onClick={() => {
				emailVerify();
			}}
		>
			Verify Your Email
		</button>
	);
};

export default App;
