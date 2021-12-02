from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, BooleanField, TextAreaField
from wtforms.validators import DataRequired, Length, EqualTo

class RegistrationForm(FlaskForm):
	username = StringField('Username', validators=[DataRequired(), Length(min=2, max=40)])
	password = PasswordField('Password', validators=[DataRequired()])
	confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
	createAccount = SubmitField('Create Account')

class LoginForm(FlaskForm):
	username = StringField('Username', validators=[DataRequired(), Length(min=2, max=40)])
	password = PasswordField('Password', validators=[DataRequired()])
	login = SubmitField('Login')

class CreateUserForm(FlaskForm):
	createEmail = StringField('Email', validators=[DataRequired(), Length(min=2, max=40)])
	createUser = SubmitField('Create User')

class PasswordForm(FlaskForm):
	changeEmail = StringField('Email', validators=[DataRequired(), Length(min=2, max=40)])
	changePass = SubmitField('Request Password')

class TicketForm(FlaskForm):
	createTitle = StringField('Title', validators=[DataRequired()])
	createDesc = TextAreaField('Description', validators=[DataRequired()])
	createTicket = SubmitField('Send Ticket')

class NoteForm(FlaskForm):
	createNoteTitle = StringField('Title', validators=[DataRequired()])
	createNoteDesc = TextAreaField('Description', validators=[DataRequired()])
	createNote = SubmitField('Add Note')